#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:15:22 2020

@author: johanna
"""


import numpy as np
import mancala as m

ma = m.Mancala(exploration_rate = 0.2)
print("Start")
print(ma.net.biases)   
print(ma.spielfeld[0:12])
#ma.train_net(500,25,1)
print("trained")
#print(ma.play())
print('play gegen Random')
matest = m.Mancala(exploration_rate = 0.2)
matest.net.load_network_from_files("Test")

Spieler1gewonnen = 0
Spieler2gewonnen = 0
unentschieden = 0
for i in range (1,2):
    #print(i)
    
    while not(np.array_equal(matest.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(matest.spielfeld[6:12] ,[0,0,0,0,0,0]) or matest.spielfeld[12]>36 or matest.spielfeld[13]>36): # muesste es nicht ausreichen zu ueberpruefen, ob die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
    
    # das geht nicht, zum Gewinn zählen noch die Bohnen auf dem Feld
    #while not(matest.spielfeld[12]>36 or matest.spielfeld[13]>36 or (matest.spielfeld[12] == 36 and matest.spielfeld[13] == 36)):
        #Spieler 1 netz
        feld = matest.get_next_action(matest.spielfeld)
        print(matest.spielfeld[feld])
        matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, feld)
        #selbst( spieler 2)
        matest.spieler1 = not matest.spieler1

        print(matest.spielfeld)
        mulde = int(input("Eingabe: int zwischen 6 und 11"))
        
        mulde  = mulde-6
        #print(mulde)
        #print(matest.spielfeld)
        matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, mulde)
        matest.spieler1 = not matest.spieler1

        #  matest.get_turned_spielfeld(matest.spielfeld)
       # print(matest.spielfeld)
    #check who won
    
    if matest.spielfeld[12] > 36:
        Spieler1gewonnen += 1
       
    elif matest.spielfeld[13] > 36:
        Spieler2gewonnen += 1
    elif matest.spielfeld[12] == 36 and matest.spielfeld[13] == 36:
        unentschieden += 1
    #else:
      #  print(matest.spielfeld[0:6])
       # print(matest.spielfeld[6:12])
    
    matest.reset()
print("Netz", Spieler1gewonnen/100, "%")
print("Random", Spieler2gewonnen/100, "%")
print("unentschieden", unentschieden/100, "%")
         