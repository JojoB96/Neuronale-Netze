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
    def spielzug1(self, feld):
        if(feld > 6):
            raise ValueError("feld ungueltig")
        
        l = self.spielfeld[feld]
        self.spielfeld[feld] =0
        
        #spielen
        for i in range(l):
            self.spielfeld[(feld+i+1) % 12] =  self.spielfeld[(feld+i+1) %12]+1
       
        #schlagen
        if (self.spielfeld[(feld+l) %12] == 2 or self.spielfeld[(feld+l) %12] == 4 or self.spielfeld[(feld+l) %12] == 6):
            self.spielfeld[12] =self.spielfeld[12]+self.spielfeld[(feld+l) %12]
            self.spielfeld[(feld+l) %12] = 0
            print("treffer")
            
    def spielzug2(self, feld):
        if(5>feld >12):
            raise ValueError("feld ungueltig")
        
        l = self.spielfeld[feld]
        self.spielfeld[feld] =0
        for i in range(l):
            self.spielfeld[(feld+i+1) % 12] =  self.spielfeld[(feld+i+1) %12]+1
       
        #schlagen
        if (self.spielfeld[(feld+l) %12] == 2 or self.spielfeld[(feld+l) %12] == 4 or self.spielfeld[(feld+l) %12] == 6):
            self.spielfeld[13] =self.spielfeld[12]+self.spielfeld[(feld+l) %12]
            self.spielfeld[(feld+l) %12] = 0
            print("treffer")
              
ma = Mancala()
print(ma.spielfeld)
ma.spielzug1(5)
print(ma.spielfeld)   
ma.spielzug2(8)       
print(ma.spielfeld)  
ma.spielzug1(4)
print(ma.spielfeld)   
ma.spielzug2(9)       
print(ma.spielfeld)   
ma.spielzug1(0)
print(ma.spielfeld)                        