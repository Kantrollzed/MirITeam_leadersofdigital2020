# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 00:04:43 2020

@author: pitonhik
"""

import bluetooth


class blut():
    time = 3
    name = ['Galaxy Buds (1CCD)']
 
    def ch_t(self,time):
        self.time = time
    def skan(self):
        devices = bluetooth.discover_devices(duration=self.time, lookup_names=True,
                                            flush_cache=True, lookup_class=False)
        return devices
    def control(self):
        d = self.skan()
        for i in range(len(d)):
            print(d[i][1])
            if d[i][1] in self.name:
                return True
        return False