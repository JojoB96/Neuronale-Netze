#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:47:20 2020

@author: johanna
"""
import numpy as np

class Mancala(object):
    def __init__(self):
        self.spielfeld = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])  #Spielfeld (zustand)
    
    
    #feld = Zahl zwischen 0 und 11
    def spielzug(self, feld):
        if(feld > 11):
            raise ValueError("feld ungueltig")
        
        l = self.spielfeld[feld]
        self.spielfeld[feld] =0
        for i in range(l):
            self.spielfeld[(feld+i+1) % 12] =  self.spielfeld[(feld+i+1) %12]+1
        
ma = Mancala()
ma.spielzug(11)       
print(ma.spielfeld)             