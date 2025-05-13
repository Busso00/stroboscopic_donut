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

**Use python3.12.3**

Setup code Linux (no stroboscopic effects):
```
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy
pip install scipy
```

Setup code Windows:
```
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
pip install numpy
pip3 install scipy
pip install PyAudioWPatch
```



Run code in full screen terminal:
```
python lb_music.py
```