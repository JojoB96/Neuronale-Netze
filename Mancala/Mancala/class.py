#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:47:20 2020

@author: johanna
"""
import numpy as np
import random

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
            #print("treffer")
            
    def spielzug2(self, feld):
        
        if(5>feld >12):
            raise ValueError("feld ungueltig")
        
        l = self.spielfeld[feld]
        self.spielfeld[feld] =0
        for i in range(l):
            self.spielfeld[(feld+i+1) % 12] =  self.spielfeld[(feld+i+1) %12]+1
       
        #schlagen
        if (self.spielfeld[(feld+l) %12] == 2 or self.spielfeld[(feld+l) %12] == 4 or self.spielfeld[(feld+l) %12] == 6):
            self.spielfeld[13] =self.spielfeld[13]+self.spielfeld[(feld+l) %12]
            self.spielfeld[(feld+l) %12] = 0
            #print("treffer")
            
    def randomspielzug1(self):
        r = random.randint(0,5)
        self.spielzug1(r)
    def randomspielzug2(self):
        r = random.randint(6,11)
        self.spielzug2(r)
              
ma = Mancala()

for i in range(200):
    if(np.array_equal(ma.spielfeld[0:6] ,[0,0,0,0,0,0])):
        print("Spiel zu ende")
        break
    ma.randomspielzug1()
    if(np.array_equal(ma.spielfeld[7:12] ,[0,0,0,0,0,0])):
        print("Spiel zu ende")
        break 
    ma.randomspielzug2()
  
   # print(ma.spielfeld)    
   
print("Spielzuege:", 2*i)
print(ma.spielfeld) 
print(np.sum(ma.spielfeld))              