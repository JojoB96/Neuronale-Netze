#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:47:20 2020

@author: johanna
"""
import numpy as np
import random
import Network
from copy import deepcopy

class Mancala(object):
    def __init__(self, exploration_rate = 0.3, network_layers = 4 , name = 'default_neurons'):
    # the Parameters are optional and are used to initialize a suitable neuronal network for the Game:
    #   exploration_rate:   scalar, determines with wich rate an random action should be done
    #   network_layers  :   scalar, contains the number of layers for the neuronal network
    #   name            :   string, is used to call the data of bias and weights 
    
        self.spielfeld          = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])  #Spielfeld (zustand)
        # self.spielfeld[12] = Schatzmulde von Spieler 1
        # self.spielfeld[13] = Schatzmulde von Spieler 2
        
        self.exploration_rate   = exploration_rate
        self.rewards            = [1.0] 
        # rewards[0]:   Das Netzwerk bekommt einen Punkt für jede Kugel die es faengt
        # rewards[.]:   *Moegliche weitere Belohnungen / Strafen*
    
        self.net                = Network.Network(network_layers, name)
        
        # Parameter for the q-function
        self.a                  = 0.3
        self.discount           = 0.4
        self.spieler1           = True
    
    
    def get_turned_spielfeld(self, spielfeld):
        new_order = [6,7,8,9,10,11,0,1,2,3,4,5,13,12]
        tmp_spielfeld = [spielfeld[i] for i in new_order]
        return tmp_spielfeld
    
    def reset(self):
        self.spielfeld          = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])
        self.spieler1           = True
    
    def check_action(self):
      #  mulden_erlaubt = np.empty([6,1])
        muldenliste = []
        for i in range(0,6):
            if self.spielfeld[i+6] > 0:
                #mulden_erlaubt[i] = True
                muldenliste.append(i)
           # else:
            #    mulden_erlaubt[i] = False
        return muldenliste        
        #return mulden_erlaubt, muldenliste
    
    def randomfeld(self):               # evtl anpassen, sodass kein Feld ausgewaehlt wird, welches keine Bohnen hat
        muldenliste = self.check_action()
        return random.choice(muldenliste)
    
        
    def guess_Q(self, spielfeld):       # Uebergebe nur spielfeld bis feld 11?
        # guess the q-value of the current spielfeld
        tmp_spielfeld = spielfeld
        if not self.spieler1:
            tmp_spielfeld = self.get_turned_spielfeld(spielfeld)
        
        return self.net.feedforward(tmp_spielfeld[0:12])
    
    
    def greedy_action(self, spielfeld):    # erwartet, dass das neuronale netz ein argument der groesse 6 ausgibt
        #choose the action that will end to the highest q-value (according to the neural network)
        gQ = self.guess_Q(spielfeld)
        while spielfeld[np.argmax(gQ)] == 0:
            gQ[np.argmax(gQ)] = 0
            #print(np.argmax(self.guess_Q(spielfeld)))
            if np.argmax(gQ) == 0:
                break
        return np.argmax(gQ)
        
        
    def get_next_action(self, spielfeld):
        if random.random() > self.exploration_rate:
            return self.greedy_action(spielfeld)
        else:
            
            return self.randomfeld()
        
        
    def get_spielfeld_and_reward_after_action(self, spielfeld, action):
        # action ist eine Zahl von 0 bis 5 und beschreibt welche Mulde der eigenen Spielfeldseite geleert wird
        
        tmp_spielfeld = spielfeld
        if not self.spieler1:
            # Wenn nicht Spieler1 am Zug ist, sondern Spieler2, dann drehe das Spielfeld um
            tmp_spielfeld  = self.get_turned_spielfeld(spielfeld)
        
        reward = tmp_spielfeld[12]
        
        # Sammel die Bohnen aus der Mulde
        bohnen = tmp_spielfeld[action]
        tmp_spielfeld[action] = 0
        
        # Verteile Bohnen auf die Mulden gegen den Uhrzeigersinn
        for b in range(bohnen):
            tmp_spielfeld[(action+b+1) % 12] =  tmp_spielfeld[(action+b+1) %12]+1
            
        # Pruefe ob eine befuellte Mulde 2, 4, oder 6 Bohnen hat
        # Ja? Dann sammel alle Bohnen aus der Mulde ein
        # Ueberpruefe dies nun im Uhrzeigersinn bis die Frage mit Nein beantwortet wird
        for b in range(bohnen):
            b = bohnen - b 
            if (tmp_spielfeld[(action+b) %12] == 2 or tmp_spielfeld[(action+b) %12] == 4 or tmp_spielfeld[(action+b) %12] == 6):
                tmp_spielfeld[12] += tmp_spielfeld[(action+b) %12]
                tmp_spielfeld[(action+b) %12] = 0
            else:
                break
        
        #Jede gefangene Bohne wird um self.rewards[0] belohnt
        reward = self.rewards[0]*(tmp_spielfeld[12] - reward)
        #...#
        if(np.array_equal(tmp_spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(tmp_spielfeld[6:12] ,[0,0,0,0,0,0])): # muesste es nicht ausreichen zu ueberpruefen, ob die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
            tmp_spielfeld[12] += sum(tmp_spielfeld[0:6])
            tmp_spielfeld[13] += sum(tmp_spielfeld[6:12])
            
        if tmp_spielfeld[12] >36:
            reward += 50
        # Hier könnte man evtl. noch überprüfen ob das Spiel gewonnen wurde und zusaetzliche Belohnung ausschuetten
        #...# unbedingt!!!
        
        if not self.spieler1:
            # Wenn nicht Spieler1 am Zug war, sondern Spieler2, dann drehe das Spielfeld zurück
            tmp_spielfeld  = self.get_turned_spielfeld(tmp_spielfeld)
        #print(tmp_spielfeld)
        return tmp_spielfeld, reward
    
    #def get_reward(self, spielfeld, action):
            
        
    def play(self):                                  # überprüfen
        Spielfeldliste = [deepcopy(self.spielfeld)]
        reward_liste   = [0.0]
        while not(np.array_equal(self.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(self.spielfeld[6:12] ,[0,0,0,0,0,0]) or self.spielfeld[12]>36 or self.spielfeld[13]>36): # muesste es nicht ausreichen zu ueberpruefen, ob die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
            feld = self.get_next_action(self.spielfeld)
            self.spielfeld, reward = self.get_spielfeld_and_reward_after_action(self.spielfeld, feld)
            Spielfeldliste.append(deepcopy(self.spielfeld))
            reward_liste.append(reward)
            self.spieler1 = not self.spieler1
        
        self.reset()
        return Spielfeldliste, reward_liste
    
    
    def create_training_data(self, spielfeld_liste, reward_liste): # unfertig
        q_liste = []
        spielfeld2 = np.empty([6,14], dtype=np.int8)
        reward1    = 0
        reward2    = 0
        q2         = np.empty([6,6])
        
        for i, r in zip(spielfeld_liste, reward_liste):
            q1 = self.guess_Q(i)
            for j in range(6):
                spielfeld2[j], reward1 = self.get_spielfeld_and_reward_after_action(i, j) # hier kann man noch probieren
                self.spieler1          = not self.spieler1
                q2[j]                  = self.guess_Q(spielfeld2[j])
                spielfeld2[j], reward2 = self.get_spielfeld_and_reward_after_action(spielfeld2[j], np.argmax(q2[j]))
                self.spieler1          = not self.spieler1
                q2[j]                  = self.guess_Q(spielfeld2[j])
                q1[j]                  = (1-self.a)*q1[j] + self.a*(reward1 + self.discount * max(q2[j]))
            q_liste.append(q1)
        return [(s[:12],q) for s,q in zip(spielfeld_liste, q_liste)]
    
    
    def train_net(self, iterations, mini_batch_length, eta):
        for i in range(iterations):
            spielfeld_liste, reward_liste = self.play()
            training_data = self.create_training_data(spielfeld_liste, reward_liste)
            
            if mini_batch_length > len(training_data):
                self.net.stochastic_update(training_data, len(training_data), eta)
            else:
                self.net.stochastic_update(training_data, mini_batch_length, eta)
            print(i)
        #print(training_data)   
        # evtl. speichern wir jedes mal / ab und zu die Gewichte und Bias (net.save_network.to_files(name))
        self.net.save_network_to_files('Test')
    
    

