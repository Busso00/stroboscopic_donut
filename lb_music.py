#import pyaudio
import numpy as np
import numpy as np
import scipy
import pyaudiowpatch as pyaudio
import time
import sys
import math

from threading import Thread, Semaphore

tick_time = [] #adjust based on T/2 of screen update

FREQ_PRINT = False
WITH_FFT = False

DIS_BASS = False
DIS_MID  = True
DIS_HIGH = False

lag = 200 #2s (>4beat @ 128bpm)
threshold_bass = 0.0 
threshold_mid = 0.0 
threshold_high = 0.0 
threshold_v = 0.0 

#2.0 hyper low use with block flesh rules
influence = 1/lag

lock = Semaphore(1)


def init(
    x,
    lag,
    threshold,
    influence,
    ):
    '''
    Smoothed z-score algorithm
    Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
    '''

    labels = np.zeros(lag)
    filtered_y = np.array(x[0:lag])
    avg_filter = np.zeros(lag)
    std_filter = np.zeros(lag)
    var_filter = np.zeros(lag)

    avg_filter[lag - 1] = np.mean(x[0:lag])
    std_filter[lag - 1] = np.std(x[0:lag])
    var_filter[lag - 1] = np.var(x[0:lag])

    return dict(avg=avg_filter[lag - 1], var=var_filter[lag - 1],
                std=std_filter[lag - 1], filtered_y=filtered_y,
                labels=labels)


def check(
    result_bass,
    single_value,
    threshold
    ):


    previous_avg = result_bass['avg']
    previous_std = result_bass['std']

    if abs(single_value - previous_avg) > threshold * previous_std:
        if single_value > previous_avg:
            return 1
        else:
            return -1
    else:
        return 0


def add(
    result_bass,
    single_value,
    lag,
    threshold,
    influence,
    ):


    previous_avg = result_bass['avg']
    previous_var = result_bass['var']
    previous_std = result_bass['std']
    filtered_y = result_bass['filtered_y'] #avg and stddev on that
    labels = result_bass['labels']

    if abs(single_value - previous_avg) > threshold * previous_std:
        if single_value > previous_avg:
            labels = np.append(labels, 1)
        else:
            labels = np.append(labels, -1)

        # calculate the new filtered element using the influence factor
        filtered_y = np.append(filtered_y, influence * single_value #single_value = new element
                               + (1 - influence) * filtered_y[-1])  #filtered_y[-1] = last added
        
        # filtered_y depend on last "memory" value + new value
        # low influence -> remember, high influence -> forgot

    else:
        labels = np.append(labels, 0)
        filtered_y = np.append(filtered_y, single_value)
    
    
    # online update formula 
    current_avg_filter = previous_avg + 1. / lag * (filtered_y[-1] - filtered_y[len(filtered_y) - lag - 1])

    # online update formula
    # E(X-Xm)**2 = E(X**2)-Xm**2
    # --> 
    # Var(old)-Var(old)+Var(new) = Var(new)
    # -->
    # Var(new) = 
    # Var(old) - 1/n*(x_i**2+x_i+1**2+...+x_j**2) + Xm(ij)**2 + 1/n*(x_i+1**2+...+x_j**2+x_j+1**2) - Xm(i+1,j+1)**2 =
    # Var(old) + 1/n* (x_j+1**2 - x_i**2) + Xm(ij)**2 - Xm(i+1,j+1)**2
     
    current_var_filter = previous_var + 1. / lag * (filtered_y[-1] ** 2 - (filtered_y[len(filtered_y) - 1 - lag]) ** 2) + previous_avg**2 - current_avg_filter**2        
        

    # calculate standard deviation for current element as sqrt (current variance)
    current_std_filter = np.sqrt(current_var_filter)

    return dict(avg=current_avg_filter, var=current_var_filter,
                std=current_std_filter, filtered_y=filtered_y[1:],
                labels=labels[1:])




def task():


    TOTAL_DURATION = 100_000_000 # number of T recordings 0.02*100=2s -> 20000s = approx 6h

    RECORD_SECONDS = 0.01

    #40hz -> 0.0

    CUT_POINT_LL = 80.0
    CUT_POINT_UL = 120.0
    CUT_POINT_H = 1600.0 

   
    CH = 1

    
    p = pyaudio.PyAudio()

    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    
    if not default_speakers["isLoopbackDevice"]:
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
        else:
            print("Default loopback output device not found 😭")
            exit()

    sampleRate = int(default_speakers['defaultSampleRate'])
    chunk = int(sampleRate * RECORD_SECONDS)

    stream = p.open(format = pyaudio.paFloat32,
        channels = CH,
        rate = sampleRate,
        input = True,
        input_device_index = default_speakers['index'],
        frames_per_buffer = chunk)


    lb, la = scipy.signal.butter(4, [CUT_POINT_LL, CUT_POINT_UL], fs=float(sampleRate), btype='bandpass')
    mb, ma = scipy.signal.butter(4, [CUT_POINT_UL,CUT_POINT_H], fs=float(sampleRate), btype='bandpass')
    hb, ha = scipy.signal.butter(8, CUT_POINT_H, fs=float(sampleRate), btype='highpass')
    


    def orig_track():
        global threshold_bass
        global threshold_mid
        global threshold_high
       
        global tick_time
        
        result_bass = init(np.zeros(lag), lag, threshold=threshold_bass, influence=influence)
        result_high = init(np.zeros(lag), lag, threshold=threshold_high, influence=influence)
        
    

        #see it as volume
        running_mean_bass = 0.0
        running_mean_mid = 0.0 
        running_mean_high = 0.0 
        running_mean = 0.0
        count = 0
        
        bpm_bass = 128/4 #fixed change color
        wait_bass = lag/(lag*RECORD_SECONDS/(60/bpm_bass))
        penalty_t_bass = wait_bass

        bpm_high = 128*16
        wait_high=lag/(lag*RECORD_SECONDS/(60/bpm_high))
        penalty_t_high = wait_high

        for _ in range(int(TOTAL_DURATION*RECORD_SECONDS)): #do this for 10 seconds
            
            indata = np.frombuffer(stream.read(chunk), dtype=np.float32)
            
            
            out_bass = scipy.signal.filtfilt(lb, la, indata, axis=0)
            out_high = scipy.signal.filtfilt(hb, ha, indata, axis=0)
            out_mid = scipy.signal.filtfilt(mb, ma, indata, axis=0)
            #denominator of derivative is constant--> high change --> note  

            bass_energy = (out_bass**2).mean()
            mid_energy = (out_mid**2).mean()
            high_energy = (out_high**2).mean()
            
            energy = (indata**2).mean()



            count += 1
            running_mean_bass += (bass_energy-running_mean_bass)/count
            running_mean_mid += (mid_energy-running_mean_mid)/count
            running_mean_high += (high_energy-running_mean_high)/count

            running_mean += (energy-running_mean)/count

            
            if wait_bass > 0:
                #sinusoidal penalty
                penalty_bass = 16*np.abs(np.cos(penalty_t_bass/wait_bass*math.pi)) #penalty at start is max -> 0 -> max
                threshold_bass = penalty_bass
            
            if wait_high > 0:
                
                threshold_high = 2.0

            
            if True:
                
                
                lock.acquire()
            
                result_bass = add(result_bass, float(bass_energy+high_energy), lag=lag, threshold=threshold_bass, influence=influence)
                result_high = add(result_high, float(energy), lag=lag, threshold=threshold_high, influence=influence)
                
                ck_bass = result_bass["labels"][-1]
                ck_high = result_high["labels"][-1] and energy > running_mean*1.5


                flash_bass = ck_bass and not DIS_BASS
                flash_high = ck_high and not DIS_HIGH

                if flash_bass:
                    tick_time.append(1)
                    penalty_t_bass = wait_bass
                else:
                    penalty_t_bass -= 0.5
                    if penalty_t_bass <= 0:
                        penalty_t_bass = wait_bass


                if flash_high:
                    tick_time.append(4)
                    penalty_t_high = wait_high
                    
                else:
                    penalty_t_high -= 0.5
                    if penalty_t_high <= 0:
                        penalty_t_high = wait_high

            
            
                

                if FREQ_PRINT:
                    print("\033[1;31m"+min(int(bass_energy*10/running_mean_bass),100)*"*"+"\033[0;0m")
                    print("\033[1;32m"+min(int(mid_energy *10/running_mean_mid),100)*"*"+"\033[0;0m")
                    print("\033[1;33m"+min(int(high_energy*10/running_mean_high),100)*"*"+"\033[0;0m")
                
                    
                lock.release()

                
    
            
    orig_track()

    stream.stop_stream()
    stream.close()
    p.terminate()



thread = Thread(target=task)

DONUT = False
DIFF = False

ncol = 7
col = [f"\033[1;3{i}m" for i in range(1,8)]


if __name__=="__main__": #example of usage

    thread.start()

    color = col[0]

    index = 0

    if not DONUT:

        
        while True:
            time.sleep(0.02) #not sufficien sometimes to put process away and run "beat marker"
            #minimum rate 0.4
            lock.acquire()

            sys.stdout.write(f"\033c") #original color
            

            if tick_time != []:
                
                if not FREQ_PRINT:

                    str_v = []


                    

                    if DIFF:
                        if 1 in tick_time:
                            str_v.append(16*"\U00002588"+"\033[0;0m\n")
                        else:
                            str_v.append("\n")

                        if 2 in tick_time:
                            str_v.append("\033[1;31m"+16*"\U00002588"+"\033[0;0m\n")
                        else:
                            str_v.append("\n")

                        if 3 in tick_time:
                            str_v.append("\033[1;32m"+16*"\U00002588"+"\033[0;0m\n")
                        else:
                            str_v.append("\n")
                        
                        if 4 in tick_time:
                            str_v.append("\033[1;34m"+16*"\U00002588"+"\033[0;0m\n")
                        else:
                            str_v.append("\n")
                    else:

                        if 1 in tick_time:
                            index = (index+1)%ncol
                        
                        if 4 in tick_time:
                            str_v.append(col[index])
                            str_v.append(4*"\U00002588"+"\n")


                    sys.stdout.write("".join(str_v)) #original color

            tick_time = [] 
            sys.stdout.flush() 
            
            lock.release()
    else:
        pass

    thread.join()
