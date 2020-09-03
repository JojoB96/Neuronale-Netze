#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:20:54 2020

@author: johanna
"""

import numpy as np
import random
import mancala as m
import matplotlib.pyplot as plt

ma = m.Mancala()
#ma.net.generate_random_network([6,12,6,2])
#input()
ma.name = 'testeinfach1'
print("Start")
#print(ma.net.biases)
print(ma.spielfeld)   
ma.print_spielfeld()
#ma.train_net(1,10,0.1,5)

print("trained")
#print(ma.play())
print('play gegen Random')
ma.net.generate_random_network([14,10,6])
ma.name = 'testeinfach1'
print("Start")
ma.print_spielfeld()
Spieler1 = np.array([0])
Spieler2 = np.array([0])
unentschieden2 = np.array([0])
l=1
for j in range(1,30):
    print(j)
    ma.train_net(100,10,l,5)
    matest = m.Mancala(exploration_rate = 0.0, name = "testeinfach1", network_layers = 3)

    Spieler1gewonnen = 0
    Spieler2gewonnen = 0
    unentschieden = 0
    
    for i in range(1,100):
        
        while not(np.array_equal(matest.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(matest.spielfeld[6:12] ,[0,0,0,0,0,0]) or matest.spielfeld[12]>36 or matest.spielfeld[13]>36): # muesste es nicht ausreichen zu ueberpruefen, ob die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
            #matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)
 #            matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)

            

           
             feld = matest.get_next_action(matest.spielfeld) 
            # matest.print_spielfeld()
           #  print("guessQ",matest.guess_Q(matest.spielfeld))
            # print(feld, matest.spielfeld[feld])
           # matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)

             matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, feld)
            #print(matest.spielfeld)

           # matest.print_spielfeld()
             
            
            
            
             matest.spieler1 = not matest.spieler1
          #  print(matest.spielfeld)
             #matest.print_spielfeld()

             muldenliste = matest.check_action()
             if not muldenliste:
                matest.spielfeld[4] += sum(matest.spielfeld[0:6])
                matest.spielfeld[5] += sum(matest.spielfeld[6:12])
                break
             mulde = random.choice(muldenliste) 
             if matest.get_turned_spielfeld(matest.spielfeld)[mulde] ==0:
                 break
          #   print("mulde",mulde)
             matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, mulde)
        
             
            
            
            
             matest.spieler1 = not matest.spieler1
          #  print(matest.spielfeld)

        
    #check who won
    
        
        if matest.spielfeld[12] > 24:
            Spieler1gewonnen += 1
        #print(matest.spielfeld)
        elif matest.spielfeld[13] > 24:
            Spieler2gewonnen += 1
        elif matest.spielfeld[12] == 24 and matest.spielfeld[13] == 24:
            unentschieden += 1
    
        matest.reset()
        
    if Spieler1gewonnen >80:
        l=0.01
        ma.a = ma.a/10
        ma.exploration_rate = 0
    elif Spieler1gewonnen >70:
        l=0.01
        ma.a = ma.a/10
        ma.exploration_rate = 0.1
    else:
        l=1
        ma.a = ma.a+0.1
        ma.exploration_rate = 0.3
        
        '''
    if Spieler2gewonnen > Spieler1gewonnen:
        ma.exploration_rate = 0.5
        ma.a = 0.2
        l = 1
        print(0.5)
    if (5/3)*Spieler2gewonnen < Spieler1gewonnen:   
        ma.exploration_rate = 0.1
        ma.a = ma.a/2
        l = l/2
        print(0.1)
        '''
    print("SPieler1", Spieler1gewonnen/1, "%")
    print("Spieler2", Spieler2gewonnen/1, "%")
    print("unentschieden", unentschieden/1, "%")
           

#matest.spieler1 = not matest.spieler1
 #            matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)

            

           
  #           feld = matest.get_next_action(matest.spielfeld) 
         #   matest.print_spielfeld()
          #  print("guessQ",matest.guess_Q(matest.spielfeld))
           # print(feld, matest.spielfeld[feld])
   #          matest.spielfeld = matest.get_turned_spielfeld(matest.spielfeld)

    #         matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, feld)
            #print(matest.spielfeld)
    Spieler1 = np.append(Spieler1,[Spieler1gewonnen/1])
    Spieler2 = np.append(Spieler2,[Spieler2gewonnen/1])
    unentschieden2 = np.append(unentschieden2,[unentschieden/1])
plt.plot(Spieler1)
plt.plot(Spieler2)
plt.plot(unentschieden)

