#import pyaudio
import numpy as np
import numpy as np
import scipy
import struct
import pyaudiowpatch as pyaudio
import time
import librosa
import time, sys 

from threading import Thread

tick_time = 0 #adjust based on T/2 of screen update

FREQ_PRINT = False
WITH_FFT = False

lag = 16
threshold = 2.0 #3.3 for all+bass+high #4.5 for high+bass #3.0 for all 
#2.0 hyper low use with block flesh rules
influence = 0.1

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
    result,
    single_value,
    threshold
    ):


    previous_avg = result['avg']
    previous_std = result['std']

    if abs(single_value - previous_avg) > threshold * previous_std:
        if single_value > previous_avg:
            return 1
        else:
            return -1
    else:
        return 0



def add(
    result,
    single_value,
    lag,
    threshold,
    influence,
    ):


    previous_avg = result['avg']
    previous_var = result['var']
    previous_std = result['std']
    filtered_y = result['filtered_y'] #avg and stddev on that
    labels = result['labels']

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
                labels=labels)


t0 = 0

last_en = np.zeros((lag*4,))
index = 0

def task():


    TOTAL_DURATION = 100_000_000 # number of T recordings 0.02*100=2s -> 20000s = approx 6h

    RECORD_SECONDS = 0.01

    CUT_POINT_L = 160.0 #160.0 for dynamic

    CUT_POINT_H = 2560.0 #2000.0 for dynamic

   
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


    lb, la = scipy.signal.butter(8, CUT_POINT_L, fs=float(sampleRate))
    hb, ha = scipy.signal.butter(8, CUT_POINT_H, fs=float(sampleRate), btype='highpass')
    mb, ma = scipy.signal.butter(4, [CUT_POINT_L,CUT_POINT_H], fs=float(sampleRate), btype='bandpass')



    def orig_track():
        global t0
        global tick_time
        
        result = init(np.zeros(lag), lag, threshold=threshold, influence=influence)

        running_mean = 0.0 #is volume
        count = 0

        
        index = 0
        
        

        for i in range(int(TOTAL_DURATION*RECORD_SECONDS)): #do this for 10 seconds
            
            indata = np.frombuffer(stream.read(chunk), dtype=np.float32)
            
            t0 = time.perf_counter()

            out_bass = scipy.signal.filtfilt(lb, la, indata, axis=0)
            out_high = scipy.signal.filtfilt(hb, ha, indata, axis=0)
            out_mid = scipy.signal.filtfilt(mb, ma, indata, axis=0)
            #denominator of derivative is constant--> high change --> note

            bass_energy = (out_bass**2).mean()
            high_energy = (out_high**2).mean()
            mid_energy = (out_mid**2).mean()

            
            energy = (indata**2).mean()

            mod_energy = energy+bass_energy+high_energy

            #last_en[index] = mod_energy
            #index = (index + 1) % (lag*2)

            count += 1
            running_mean += (mod_energy-running_mean)/count

            if FREQ_PRINT:
                print("\033[1;31m"+min(int(bass_energy*10/running_mean),100)*"*"+"\033[0;0m")
                print("\033[1;32m"+min(int(mid_energy *10/running_mean),100)*"*"+"\033[0;0m")
                print("\033[1;33m"+min(int(high_energy*10/running_mean),100)*"*"+"\033[0;0m")
            

            ck = check(result, float(mod_energy), threshold=threshold) 
            #just check if taken add higher value, if not add itself
            
           
            strobo_active = False
            
            if mod_energy > 0: #make goog smoothing in general
                strobo_active = True
                bass_s = int(-lag//2)
                mid_s = int(-lag//2)
                high_s = int(-lag//2)
            
            #lag*0.01 = 0.16 
            #@128 bpm approx 2bs 1b/0.5s --> 8 time division 0.6125
            #bass at @ > td/2 -> 0.25, mid and high @ > td/4

            Me = max(bass_energy*2, mid_energy, high_energy*2)

            

            if strobo_active:
                
                past_bass = result["labels"][bass_s:-1][result["labels"][bass_s:-1] == 1]
                past_mid = result["labels"][mid_s:-1][result["labels"][mid_s:-1] == 2]
                past_high = result["labels"][high_s:-1][result["labels"][high_s:-1] == 3]

                
                flash = (past_bass.shape[0] == 0 and bass_energy*2 == Me) or ( past_mid.shape[0] == 0 and mid_energy == Me) or (past_high.shape[0] == 0 and high_energy*2 == Me) 
                flash = flash and np.count_nonzero(result["labels"][-3:-1] > 0) == 0

            else:

                flash = False

            if (ck > 0) and flash and strobo_active: #for strobo light only if volume is high are active
                #artificial peak
                #between 5*10**7 - 2*10**8 and
                result = add(result, float(mod_energy*5*10**7), lag=lag, threshold=threshold, influence=influence)

                

                if not FREQ_PRINT: #don't update tick to not make other process print
                    
                    
                    if Me == bass_energy*2:
                        tick_time = 1
                    elif Me == mid_energy:
                        tick_time = 2
                    elif Me == high_energy*2:
                        tick_time = 3

                
                #if detected multiply input value (want only peaks)
            
            else:
                result = add(result, float(mod_energy), lag=lag, threshold=threshold, influence=influence)


            if ck > 0:

                
                if Me == bass_energy*2:
                    result["labels"][-1] = 1
                elif Me == mid_energy:
                    result["labels"][-1] = 2
                elif Me == high_energy*2:
                    result["labels"][-1] = 3

               
                 

                
    #     
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    def tempo_track(): #this is slow #fft + other postproccess (do once in a while)
        
        print("tempo_track not use")
        exit(-1)

        global tick_time

        running_mean = 0.0 #for volume
        count = 0


        CB_LEN = 400 #2s buffer for tempo tracking
        index = 0
        
        last = np.zeros((CB_LEN*sampleRate,)) #for online tempo tracking

        took_sliding_window = [] #keep track of already taken
        last_sliding_window = []

        took2_sliding_window = []
        last2_sliding_window = []

        sliding_window_size = 10
        sliding_window_size2 = sliding_window_size//2

        for i in range(int(TOTAL_DURATION*RECORD_SECONDS)):
            
            indata = np.frombuffer(stream.read(chunk), dtype=np.float32)
            
            
            out_bass = scipy.signal.filtfilt(lb, la, indata, axis=0)
            out_high = scipy.signal.filtfilt(hb, ha, indata, axis=0)
            #out_mid = scipy.signal.filtfilt(mb, ma, indata, axis=0)
            #denominator of derivative is constant--> high change --> note
            #bass_norm = np.abs(out_bass).mean()
            #high_norm = np.abs(out_high).mean()
            #mid_norm = np.abs(out_mid).mean()

            #print(bass_norm, mid_norm, high_norm)
            
            norm = np.abs(indata).mean()
            
            norm2 = ((out_bass**2).mean()+(out_high**2).mean())*norm

            count += 1
            running_mean += (norm-running_mean)/count

            if i % (CB_LEN*50) < CB_LEN: #at least cb_len elements
                last[index*chunk:(index+1)*chunk] = indata
            
            index = (index+1)%CB_LEN

            
            if i % (CB_LEN*50) == CB_LEN: #at least cb_len elements
                print("sync")
                
                #following ops take a while, do sometimes
                #prior = scipy.stats.uniform(80, 180) #don't use
                env = librosa.onset.onset_strength(y=last, sr=float(sampleRate))
                utempo = librosa.feature.tempo(onset_envelope=env, sr=float(sampleRate), max_tempo=180.0) #regulate length of array (of past beat), BPM

                bps = (utempo/60) #b/s
                #second every each there is a beat
                bT = 1/bps
                sT = RECORD_SECONDS
                

                #I want to record one entire sliding window without a beat -> sliding_window size is 

                sliding_window_size = int(bT/sT/4) #do "phase locking"
                sliding_window_size2 = int(bT/sT/8) 

                #i want to estimate 5 with 120 bpm

                print(f"tempo: {utempo}")
                print(sliding_window_size)


            strobo_active = True

            t1 = sum(last_sliding_window)/sliding_window_size*0.6
            t2 = sum(last2_sliding_window)/sliding_window_size2*0.4

            if t1 > t2:
                cond = sum(took_sliding_window) == 0
            else:
                cond = sum(took2_sliding_window) == 0
            
            if strobo_active and cond and norm2 > (t1 + t2)*2:
                
                took_sliding_window.append(1)
                last_sliding_window.append(norm2)
                
                took2_sliding_window.append(1)
                last2_sliding_window.append(norm2)

                tick_time = 1
            else:

                took_sliding_window.append(0)
                last_sliding_window.append(norm2)

                took2_sliding_window.append(0)
                last2_sliding_window.append(norm2)


            if (sliding_window_size < len(last_sliding_window)):
                took_sliding_window = took_sliding_window[len(last_sliding_window)-sliding_window_size:-1]
                last_sliding_window = last_sliding_window[len(last_sliding_window)-sliding_window_size:-1]

                took2_sliding_window = took2_sliding_window[len(last2_sliding_window)-sliding_window_size:-1]
                last2_sliding_window = last2_sliding_window[len(last2_sliding_window)-sliding_window_size:-1]
            else:
                for i in range(sliding_window_size, len(last_sliding_window)):
                    
                    took_sliding_window.append(norm2)
                    last_sliding_window.append(0.0)

                    if i % 2 ==0:
                        took2_sliding_window.append(norm2)
                        last2_sliding_window.append(0.0)


    #tempo_track()
    orig_track()

    stream.stop_stream()
    stream.close()
    p.terminate()



thread = Thread(target=task)

thread.start()

pos = 0

while True:
    time.sleep(0.04) #not sufficien sometimes to put process away and run "beat marker"
    #minimum rate 0.4
    if tick_time > 0:
        if not FREQ_PRINT:
            if tick_time == 1:
                print("\033[1;31m"+pos*"*"+"\033[0;0m")
            elif tick_time == 2:
                print("\033[1;32m"+pos*"*"+"\033[0;0m")
            elif tick_time == 3:
                print("\033[1;33m"+pos*"*"+"\033[0;0m")
        pos = (pos+1)%50
        tick_time = 0 # process can double run into it beside sleep

    else:
        print("\033c")
        

thread.join()