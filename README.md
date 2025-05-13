# STROBOSCOPIC_DONUT :coffee::doughnut:
A stroboscopic donut 

It's not actually the simplest thing I have implemented ;)

It require to know:
- Calculus 2,
- Numpy,
- Signal processing (filters, energy of a signal),
- Other algorithm to do peak detection in realtime data (z-score) 
  
...and have a bit of creativity to put together all this stuff for a good result

References:
- Donut math (only toroid): https://www.a1k0n.net/2011/07/20/donut-math.html 
- Music tempo detection: https://www.youtube.com/watch?v=FmwpkdcAXl0&list=LL&index=7
- z-score algorithm: https://github.com/MatteoBattilana/robust-peak-detection-algorithm

Setup code:
```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
pip install scipy
pip install PyAudioWPatch
```

Run code in full screen terminal:
```
python lb_music.py
```