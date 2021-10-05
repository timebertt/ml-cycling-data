# ml-cycling-data

Term paper for the lecture "Machine Learning" at DHBW CAS.

## Jupyter Lab

Start Jupyter Lab:

```bash
$ docker-compose up -d
...

$ docker-compose logs
...
jupyter_1  |     To access the server, open this file in a browser:
jupyter_1  |         file:///home/jovyan/.local/share/jupyter/runtime/jpserver-7-open.html
jupyter_1  |     Or copy and paste one of these URLs:
jupyter_1  |         http://83875c4de56e:8888/lab?token=xxxxxxxxxxxxxxxxxxx
jupyter_1  |      or http://127.0.0.1:8888/lab?token=xxxxxxxxxxxxxxxxxxx
```

Click / open `http://127.0.0.1:8888/lab?token=xxxxxxxxxxxxxxxxxxx` to open Jupyter Labs.
Directory `jupyter` will be mounted and visible in Jupyter Labs as `work`.

Stop Jupyter Lab:
```bash
$ docker-compose down
...
```
