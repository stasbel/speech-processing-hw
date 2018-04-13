# task03

## Prereqs

`nvidia-docker`, a free port and VCTK data.

## Running

Assuming your have free port `6167` and your VCTK data at `/media/VCTK-Corpus`, your operates as follows:

```
nvidia-docker build -t  deepspeech2.docker .
nvidia-docker run -it -v /media/VCTK-Corpus:/app/data -v `pwd`:/app/code -p 6167:8888 deepspeech2.docker
```

You can now access jupyter notebook at specified port.