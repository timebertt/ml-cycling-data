# Programmentwurf Maschinelles Lernen

[Zur Seminararbeit](./paper-final.pdf)

[Zum Jupyter Notebook](./jupyter/cycling_data_final.ipynb)

## Info

Diese Seminararbeit wurde von Nikola Braukmüller und Tim Ebert als Prüfungsleistung im Modul "Maschinelles Lernen und Computational Intelligence" im Rahmen des Masterstudiums in Informatik am [DHBW Center for Advanced Studies](https://www.cas.dhbw.de/) (CAS) verfasst.

Copyright (c) 2021 Nikola Braukmüller und Tim Ebert

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
