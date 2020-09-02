# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:46:33 2020

@author: Johanna
"""


import numpy as np
import random
import testspiel as t

ma = t.Testspiel()
ma.net.generate_random_network([14,12,23,6])
#input()
ma.name = 'testeinfach'
print("Start")
#print(ma.net.biases)   
ma.print_spielfeld()
for j in range(1,50): 
    ma.train_net(10,10,1/j,5)
    #print("trained")
    #print(ma.play())
    #print('play gegen Random')
    matest = t.Testspiel(exploration_rate = 0.0, name = "testeinfach", network_layers = 4)

    #lasse Netz als Spieler 1 gegen Random spielen
    Spieler1gewonnen = 0
    Spieler2gewonnen = 0
    unentschieden = 0

    for i in range (1,1000):
    #print(i)
        while not(np.array_equal(matest.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(matest.spielfeld[6:12] ,[0,0,0,0,0,0]) or matest.spielfeld[12]>36 or matest.spielfeld[13]>36): # muesste es nicht ausreichen zu ueberpruefen, ob die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
        #print(matest.spielfeld)
    # das geht nicht, zum Gewinn zählen noch die Bohnen auf dem Feld
    #while not(matest.spielfeld[12]>36 or matest.spielfeld[13]>36 or (matest.spielfeld[12] == 36 and matest.spielfeld[13] == 36)):
        #Spieler 1 netz
        
           # matest.print_spielfeld()
           # print("Spieler 1")
            #print(matest.guess_Q(matest.spielfeld))
            #######################################
            # matest.print_spielfeld()
            #print("Spieler 2")
            #print(matest.guess_Q(matest.spielfeld))
        
            #########################################
            feld = matest.get_next_action(matest.spielfeld)
            #feld = np.argmax(matest.guess_Q(matest.spielfeld))
           # print(feld)
            matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, feld)
            #  matest.print_spielfeld()

            #print("reward",reward)
            #########################
            matest.spieler1 = not matest.spieler1
            #############################
        
          
            muldenliste = matest.check_action()
            if not muldenliste:
                matest.spielfeld[12] += sum(matest.spielfeld[0:6])
                matest.spielfeld[13] += sum(matest.spielfeld[6:12])
            # hier könnte mann noch die restlichen Felder auf Null setzen
                break

            mulde = random.choice(muldenliste)
        
            matest.spielfeld, reward = matest.get_spielfeld_and_reward_after_action(matest.spielfeld, mulde)

       
        
       
        
            matest.spieler1 = not matest.spieler1
        
               #random spieler 2
       

        #check who won
    
        if matest.spielfeld[12] > 10:
            Spieler1gewonnen += 1
        #print(matest.spielfeld)
        elif matest.spielfeld[13] > 10:
            Spieler2gewonnen += 1
        elif matest.spielfeld[12] == 10 and matest.spielfeld[13] == 10:
            unentschieden += 1
       
    
        matest.reset()
    print("Spieler1", Spieler1gewonnen/10, "%")
    print("Spieler2", Spieler2gewonnen/10, "%")
    print("unentschieden", unentschieden/10, "%")
           
