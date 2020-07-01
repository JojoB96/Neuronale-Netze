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
import itertools

class Mancala(object):
    def __init__(self, exploration_rate = None, network_layers = 4 , name = 'default_neurons'):
    # the Parameters are optional and are used to initialize a suitable neuronal network for the Game:
    #   exploration_rate:   scalar, determines with wich rate an random action should be done
    #   network_layers  :   scalar, contains the number of layers for the neuronal network
    #   name            :   string, is used to call the data of bias and weights 
    
      ##  if exploration_rate is None or network_layers is None or name is None:
        #    self.spielfeld          = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])  #Spielfeld (zustand)
         #   print("Watch out! Due to missing parameters a neuronal network is not initialized yet!")
        #else:
            self.spielfeld          = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])  #Spielfeld (zustand)
            self.exploration_rate   = exploration_rate
            self.rewards            = [1.0] 
            # rewards[0]:   Das Netzwerk bekommt einen Punkt für jede Kugel die es faengt
            # rewards[.]:   *Moegliche weitere Belohnungen / Strafen*
    
            self.net                = Network.Network(network_layers, name)
        
            # Parameter for the q-function
            self.a                  = 0.3
            self.discount           = 0.4
            self.spieler1           = True
    
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
        return self.spielfeld
        
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
        return(self.spielfeld)
        
    def randomfeld(self):
        if self.spieler1 == True:
            r = random.randint(0,5)
            
        else:
            r = random.randint(6,11)
        return r
        
    def guess_Q(self, spielfeld):
        # guess the q-value of the current spielfeld
        return self.net.feedforward(spielfeld)
    
    def greedy_action(self, spielfeld):
        #choose the action that will end to the highest q-value (according to the neural network)
        if self.spieler1 == True:
            return np.argmax(self.guess_Q(spielfeld))
        else:
            return np.argmax(self.guess_Q(spielfeld))+7
        
    
    
    def get_next_action(self, spielfeld):
        if random.random() > self.exploration_rate:
            return self.greedy_action(spielfeld)
        else:
            return self.randomfeld()          #  Hier müssen wir uns nochmal genau überlegen, wo wir die regeln einbauen wollen
     
    def take_action(self, feld):
        if self.spieler1 == True:
            self.spieler1 = False
            return(self.spielzug1(feld))
        else:
            self.spieler1 = True
            return(self.spielzug2(feld))
            
        
    def play(self):
        Spielfeldliste = [deepcopy(self.spielfeld)]
        while not(np.array_equal(self.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(self.spielfeld[7:12] ,[0,0,0,0,0,0])): # muesste es nicht ausreichen zu ueberpruefen, pb die schatzmulden mehr als die haelfte der Kugeln beinhalten? ( also self.spielfeld[12] > 36 or also self.spielfeld[13] > 36
            feld = self.get_next_action(self.spielfeld)
            self.spielfeld = self.take_action(feld)
            Spielfeldliste.append(deepcopy(self.spielfeld))
        return Spielfeldliste
    
    
    def create_training_data(self, spielfeld_liste):
        q_liste = []
        for i in spielfeld_liste:
            q = self.guess_Q(i)
            q_liste.append(q)
        return [(s,q) for s,q in zip(spielfeld_liste, q_liste)]
    
    
    def train_net(self, iterations, mini_batch_length, eta):
        for i in range(iterations):
            spielfeld_liste = self.play()
            training_data = self.create_training_data(spielfeld_liste)
            if mini_batch_length > len(spielfeld_liste):
                self.net.stochastic_update(spielfeld_liste, len(spielfeld_liste), eta)
            else:
                self.net.stochastic_update(spielfeld_liste, mini_batch_length, eta)
            
        print(training_data)   
        # evtl. speichern wir jedes mal / ab und zu die Gewichte und Bias (net.save_network.to_files(name))
    
    
ma = Mancala(exploration_rate = 0.4)
print("Start")
print(ma.net.biases)       
ma.train_net(5,20,1)
print(ma.net.biases)
