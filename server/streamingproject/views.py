from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError , HttpResponse
from django.core import serializers
from django.views.decorators import gzip
import cv2
import os
import sys
import json
from videoProccesing.openvino_processing import ImageOpenVINOPreprocessing
#from videoProccesing.eye import *
ImgProcess = ImageOpenVINOPreprocessing()
'''eyeris_detector = EyerisDetector(image_source=ImageSource(), classifier=CascadeClassifier(),
                                 tracker=LucasKanadeTracker())'''

text = ''
blut = ''
click = 0
programs = ''
glaza = 0
rez ='good stydent'
gl = ''
class VideoCamera(object):

    def __init__(self, path):
        self.video = cv2.VideoCapture(0)
        self.score = 0

        camx, camy = [(1920, 1080), (1280, 720), (800, 600), (480, 480)][1]  # Set camera resolution [1]=1280,720
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, camx)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, camy)

        self.ROOT_DIR = os.path.abspath("")
        print(self.ROOT_DIR)

        sys.path.append(self.ROOT_DIR)

        self.codec = cv2.VideoWriter_fourcc(*'DIVX')
        print("\nVizualize:")

    def __del__(self):
        self.video.read()

    # Обработчик фрейма
    def get_frame(self):
        ret, frame = self.video.read()
        if ret:
            frame = ImgProcess.main(frame)
            #frame = eyeris_detector.run(frame)
            jpeg = cv2.imencode('.jpg', frame)[1].tostring()
            return jpeg


# Обработчик камеры
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def indexscreen(request):
    global text
    try:
        template = "screens.html"
        
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")

def changeline(request):
    global text
    global blut 
    global click 
    global programs 
    global glaza 
    global rez 
    global gl
    print('-------------------------------------------')
    #print(request)
    rec = str(request).split('/')
    mes = rec[2].replace('%20',' ')
    rec = mes
    text = mes
    if 'press' in rec:
        click = click+1
    elif 'start program:' in rec:
        prog = rec.split(':')
        prog=prog[1]
        programs=prog
    elif 'bluetuse detection:' in rec:
        prog = rec.split(':')
        prog=prog[1]
        blut=prog
    elif 'eye:' in rec:
        r = rec.split(':')
        rr =r[1]
        if gl!=rr:
         glaza=glaza+1
        
        gl = rr
        
    else:
        print('not info')
    if glaza > 5:
        rez='bad student'
    print(glaza)
    print('-------------------------------------------')
    
def getline(request):
    global text
    global blut 
    global click 
    global programs 
    global glaza 
    global rez 
    response_data = {}
    response_data['text'] = text
    response_data['blut'] = blut
    response_data['click'] = click
    response_data['programs'] = programs
    response_data['glaza'] = glaza
    response_data['rez'] = rez
    return HttpResponse(json.dumps(response_data), content_type="application/json")
@gzip.gzip_page
def dynamic_stream(request, num=0,stream_path="0"):
   
    try:
        return StreamingHttpResponse(gen(VideoCamera(stream_path)), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("aborted");