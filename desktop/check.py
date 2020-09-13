import psutil # pip install psutil
import os
import requests as re
import time
class internet(object):
    def __init__(self):
     self.myhost = 'http://127.0.0.1:8000/stream/screen/'
     self.chechost = 'https://yandex.ru/'
    def internet_check(self,url):
     try:
        r = re.get(url)
        if r.status_code==200:
            return True
        else:
            return False
     except:
        return False
        
    def check(self):
        x = self.internet_check(self.myhost)
        y = self.internet_check(self.chechost)
        if x and y:
            print('good internet')
            return 2
        elif y:
            print('no conection server')
            return 1
        else:
            print('no internet')
            return 0
class program(object):
    def __init__(self):
        self.proc_name = ['chrome.exe','Discord.exe','steam.exe', 'obs64.exe']
        self.prog = {}
        self.clean()
    def clean(self):
       for i in range(len(self.proc_name)):
          self.prog[self.proc_name[i]]=0
    def check(self):
        for proc in psutil.process_iter():
           for i in self.proc_name:
              if proc.name() == i:
                 #
                 self.prog[i]=1
                 #os.system("taskkill /f /im "+i)
        for i in self.proc_name:
            if self.prog[i]==1:
                print ("Process {}  started".format(i))
        return self.prog
    
            
class kontroler(object):
    def __init__(self):
        self.internet= internet()
        self.program=program()
    def send(self,info):
        re.get()
    def start(self):
        i =self.internet.check()
        p = self.program.check()
        print(i)
        print(p)
        
k = kontroler()
k.start()       


 
 

            