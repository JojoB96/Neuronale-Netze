#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:20:54 2020

@author: johanna
"""

import numpy as np
import random
import mancala as m

matest = m.Mancala(exploration_rate = 0.3, name = "Test", network_layers = 4)

#lasse Netz als Spieler 1 gegen Random spielen
Spieler1gewonnen = 0
Spieler2gewonnen = 0
unentschieden = 0

k = 50000
for i in range (1,k):
    #print(i)
    spielfelder = []
    
    while not(np.array_equal(matest.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(matest.spielfeld[6:12] ,[0,0,0,0,0,0]) or matest.spielfeld[12]>36 or matest.spielfeld[13]>36): 
        #Spieler 1 netz
        
        feld = matest.get_next_action(matest.spielfeld)
        matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, feld)
        
        spielfelder.append(matest.spielfeld)
        
        #random spieler 2
        matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)
        muldenliste = matest.check_action()
        if not muldenliste:
            matest.spielfeld[12] += sum(matest.spielfeld[0:6])
            matest.spielfeld[13] += sum(matest.spielfeld[6:12])
            # hier könnte mann noch die restlichen Felder auf Null setzen
            break

        mulde = random.choice(muldenliste)
        matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)
        matest.spieler1 = not matest.spieler1
        matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, mulde)
        matest.spieler1 = not matest.spieler1
        
        

    #check who won
    
    if matest.spielfeld[12] > 36:
        Spieler1gewonnen += 1
        matest.train_net2(spielfelder, 1.0, 15, 0.5)
        #print(matest.spielfeld)
    elif matest.spielfeld[13] > 36:
        Spieler2gewonnen += 1
        matest.train_net2(spielfelder, -1.0, 15, 0.5)
    elif matest.spielfeld[12] == 36 and matest.spielfeld[13] == 36:
        unentschieden += 1
       
    print(matest.guess_Q(matest.spielfeld))
    matest.reset()
print("Netz", Spieler1gewonnen/k*100, "%")
print("Random", Spieler2gewonnen/k*100, "%")
print("unentschieden", unentschieden/k*100, "%")
           
