#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:20:54 2020

@author: johanna
"""

import numpy as np
import random
import mancala as m

ma = m.Mancala()
ma.net.generate_random_network([6,6,2])
#input()
ma.name = 'test3'
print("Start")
#print(ma.net.biases)
print(ma.spielfeld)   
ma.print_spielfeld()
ma.train_net(50,50,0.1,5)

print("trained")
#print(ma.play())
print('play gegen Random')
matest = m.Mancala(exploration_rate = 0.0, name = "test3", network_layers = 3)

#lasse Netz als Spieler 1 gegen Random spielen
Spieler1gewonnen = 0
Spieler2gewonnen = 0
unentschieden = 0
for i in range (1,20):
    #print(i)
    Spieler1gewonnen = 0
    Spieler2gewonnen = 0
    unentschieden = 0
    ma.train_net(5,50,0.1,5)


    matest = m.Mancala(exploration_rate = 0.0, name = "test3", network_layers = 3)
    for j in range(1,100):
        
        while not(np.array_equal(matest.spielfeld[0:2] ,[0,0]) or np.array_equal(matest.spielfeld[2:4] ,[0,0]) or matest.spielfeld[4]>4 or matest.spielfeld[5]>4): # muesste es nicht ausreichen zu ueberpruefen, ob die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
        #print(matest.spielfeld)
    # das geht nicht, zum Gewinn zählen noch die Bohnen auf dem Feld
    #while not(matest.spielfeld[12]>36 or matest.spielfeld[13]>36 or (matest.spielfeld[12] == 36 and matest.spielfeld[13] == 36)):
        #Spieler 1 netz
        
        
        #print("Spieler 2")
        #print(matest.guess_Q(matest.spielfeld))
            
        #random spieler 2
            
            muldenliste = matest.check_action()
            if not muldenliste:
                matest.spielfeld[4] += sum(matest.spielfeld[0:2])
                matest.spielfeld[5] += sum(matest.spielfeld[2:4])
                break
            mulde = random.choice(muldenliste) 
            matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, mulde)
       # matest.print_spielfeld()
        
            matest.spieler1 = not matest.spieler1
           
      #  print("Spieler 1")
       # print(matest.guess_Q(matest.spielfeld))
            feld = matest.get_next_action(matest.spielfeld)  
            matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, feld)
          
           # matest.print_spielfeld()
            matest.spieler1 = not matest.spieler1

        
    #check who won
    
        if matest.spielfeld[4] > 4:
            Spieler1gewonnen += 1
        #print(matest.spielfeld)
        elif matest.spielfeld[5] > 4:
            Spieler2gewonnen += 1
        elif matest.spielfeld[4] == 4 and matest.spielfeld[5] == 4:
            unentschieden += 1
       
    
        matest.reset()
    print("SPieler1", Spieler1gewonnen/1, "%")
    print("Spieler2", Spieler2gewonnen/1, "%")
    print("unentschieden", unentschieden/1, "%")
           
