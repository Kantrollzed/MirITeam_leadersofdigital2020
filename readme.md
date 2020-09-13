# MirITeam
## ЦИФРОВОЙ ПРОРЫВ 2020
### РОСТЕЛЕКОМ
&nbsp;

Сервис представляет собой стриминговый сервер с десктопным клиентом для анализа 
подготовки экзаменуемого путем контроля проведения экзамена. 
С помощью компьютерного зрения, видеоаналитики и анализа исполняемых 
процессов компьютера проводится контроль действий пользователя 
во время экзамена.
На сервисе проводится контроль параметров студента 
(положение головы, направление взгляда, положение зрачков), 
процессов компьютера для проверки работы дополнительных ресурсов, 
перехват кликов и скроллинга.
Стек: Python, Django, OpenCV, OpenVINO, MobileNet, Scipy, psutil, pynput.

&nbsp;



#### Директории:

`./desktop` содержит:
 - `blut.py` - анализ соединений bluetooth
 - `check.py` - анализ интернет соединения и текущих процессов
 - `tk.py` - анализ событий мышы


 `./server` содержит:
 - `/streamingproject` - django app, транслирующий видеопоток
 - `/videoProccesing` - обработчик изображения

&nbsp;

 
#### Установка

OpenVINO:
https://docs.openvinotoolkit.org/latest/index.html
&nbsp;

&nbsp;

#### Подключение проекта:
```
cd ./your_dir_project
git clone https://github.com/Kantrollzed/MirITeam_leadersofdigital2020.git
cd ./MirITeam_leadersofdigital2020

pip install scipy, opencv-python, opencv-contrib-python, pynput, psutil, bluetooth, Django
```

&nbsp;

#### Запуск проекта:

Активация **OpenVINO**:
 - Linux: 
 ```
 source /opt/intel/openvino/bin/setupvars.sh
```
 - Windows: 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```


Запуск сервера и трансляция видео по пути `http://127.0.0.1:8000/stream/screen/` 
```
cd ./your_dir_project/MirITeam_leadersofdigital2020
python manage.py runserver
```


&nbsp;

&nbsp;
