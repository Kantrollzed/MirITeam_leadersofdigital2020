from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponseServerError

from django.views.decorators import gzip
import cv2
import os
import sys
from videoProccesing.openvino_processing import ImageOpenVINOPreprocessing


class VideoCamera(object):

    def __init__(self, path):
        self.ImgProcess = ImageOpenVINOPreprocessing()

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
            frame = self.ImgProcess.main(frame)
            jpeg = cv2.imencode('.jpg', frame)[1].tostring()
            return jpeg


# Обработчик камеры
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def indexscreen(request):
    try:
        template = "screens.html"
        return render(request,template)
    except HttpResponseServerError:
        print("aborted")

@gzip.gzip_page
def dynamic_stream(request, num=0,stream_path="0"):
    try:
        return StreamingHttpResponse(gen(VideoCamera(stream_path)), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("aborted");