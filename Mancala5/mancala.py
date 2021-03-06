#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:47:20 2020

@author: johanna
"""

# public libraries
import numpy as np
import random
#from progressbar import progressbar

from copy import deepcopy



# own libraries
import Network

class Mancala(object):
    def __init__(self, exploration_rate = 0.3, network_layers = 4 , name = 'default_neurons', name2 = "default_neurons"):
    # the Parameters are optional and are used to initialize a suitable neuronal network for the Game:
    #   exploration_rate:   scalar, determines with wich rate an random action should be done
    #   network_layers  :   scalar, contains the number of layers for the neuronal network
    #   name            :   string, is used to call the data of bias and weights 
    
        self.spielfeld          = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])  #Spielfeld (zustand)
        # self.spielfeld[12] = Schatzmulde von Spieler 1
        # self.spielfeld[13] = Schatzmulde von Spieler 2
        
        self.exploration_rate   = exploration_rate
        self.rewards            = [1, 10] 
        # rewards[0]:   Das Netzwerk bekommt einen Punkt für jede Kugel die es faengt
        # rewards[.]:   *Moegliche weitere Belohnungen / Strafen*
    
        self.net                = Network.Network(network_layers, name, "sigmoid")
        self.name               = name
        
        self.net2               = Network.Network(network_layers, name2, "sigmoid")
        self.name2              = name2
        
        # Parameter for the q-function
        self.a                  = 0.1
        self.discount           = 0.99 
        self.spieler1           = True
        self.turn               = [6,7,8,9,10,11,0,1,2,3,4,5,13,12]
    
    
    def get_turned_spielfeld(self, spielfeld):
        tmp_spielfeld = [spielfeld[i] for i in self.turn]
        return tmp_spielfeld
    
    def reset(self):
        self.spielfeld          = np.array([6,6,6,6,6,6,6,6,6,6,6,6,0,0])
        self.spieler1           = True
    
    def check_action(self, spielfeld = None):
        tmp_spielfeld = deepcopy(spielfeld)
        if spielfeld is None:
            tmp_spielfeld = deepcopy(self.spielfeld)
        if not self.spieler1:
            tmp_spielfeld = self.get_turned_spielfeld(tmp_spielfeld)
        
            
        muldenliste = []
            
        for i in range(0,6):
            if tmp_spielfeld[i] > 0:
                muldenliste.append(i)
            
        return muldenliste 
    
    
    def randomfeld(self, spielfeld = None):               # evtl anpassen, sodass kein Feld ausgewaehlt wird, welches keine Bohnen hat
        muldenliste = self.check_action(spielfeld)
        
        return random.choice(muldenliste)
    
        
    def guess_Q(self, spielfeld):       # Uebergebe nur spielfeld bis feld 11?
        # guess the q-value of the current spielfeld
        tmp_spielfeld = deepcopy(spielfeld)
        
        gQ    = self.net.feedforward(np.reshape(tmp_spielfeld,(14,1)))
        
        return gQ
    
    
    def greedy_action(self, spielfeld):    # erwartet, dass das neuronale netz ein argument der groesse 6 ausgibt
        #choose the action that will end to the highest q-value (according to the neural network)
        gQ    = self.guess_Q(spielfeld)
        
        legal = self.check_action(spielfeld)
        x     = np.zeros((len(gQ),1))
        
        np.put(x, legal, np.ones(len(legal)))
        gQ    = np.multiply(gQ, x)
        
        if self.net.act_func is Network.HyperbolicTangent:
            x = x - np.ones((len(gQ),1))
            gQ = gQ + x
        return np.argmax(gQ)
        
        
    def get_next_action(self, spielfeld):
        if random.random() > self.exploration_rate:
            return self.greedy_action(spielfeld)
        else:
            return self.randomfeld(spielfeld)
        
        
    def get_spielfeld_and_reward_after_action(self, spielfeld, action):
        # action ist eine Zahl von 0 bis 5 und beschreibt welche Mulde der eigenen Spielfeldseite geleert wird
        
        tmp_spielfeld = deepcopy(spielfeld)
        if not self.spieler1:
            # Wenn nicht Spieler1 am Zug ist, sondern Spieler2, dann drehe das Spielfeld um
            tmp_spielfeld  = self.get_turned_spielfeld(tmp_spielfeld)
        
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
            tmp_spielfeld[0:6]  = [0,0,0,0,0,0]
            tmp_spielfeld[6:12] = [0,0,0,0,0,0]
        
        
        if tmp_spielfeld[12] >36:
            tmp_spielfeld[12] += sum(tmp_spielfeld[0:6]) + sum(tmp_spielfeld[6:12])
            tmp_spielfeld[0:6]  = [0,0,0,0,0,0]
            tmp_spielfeld[6:12] = [0,0,0,0,0,0]
            reward += 10#self.rewards[1]
            
        elif tmp_spielfeld[13] > 36:
            tmp_spielfeld[13] += sum(tmp_spielfeld[0:6]) + sum(tmp_spielfeld[6:12])
            tmp_spielfeld[0:6]  = [0,0,0,0,0,0]
            tmp_spielfeld[6:12] = [0,0,0,0,0,0]
            reward -= 10
        if tmp_spielfeld[12] ==36 :
            reward +=5
        # Hier könnte man evtl. noch überprüfen ob das Spiel gewonnen wurde und zusaetzliche Belohnung ausschuetten
        #...# unbedingt!!!
        
        
        # Dies muesste man evtl rausnehmen #
        if not self.spieler1:
            # Wenn nicht Spieler1 am Zug war, sondern Spieler2, dann drehe das Spielfeld zurück
            tmp_spielfeld  = self.get_turned_spielfeld(tmp_spielfeld)
        #----------------------------------#
        
        if bohnen is 0:
            reward = -0.1
        
        #print(tmp_spielfeld)
        return tmp_spielfeld, reward
    
        
    def play(self):                                  # überprüfen
        Spielfeldliste = [deepcopy(self.spielfeld)]
        reward_liste   = [0.0]
        while not(np.array_equal(self.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(self.spielfeld[6:12] ,[0,0,0,0,0,0]) or self.spielfeld[12]>36 or self.spielfeld[13]>36):
            feld                   = self.get_next_action(self.spielfeld)
            self.spielfeld, reward = self.get_spielfeld_and_reward_after_action(self.spielfeld, feld)
            if self.spieler1:
                #Spielfeldliste.append(deepcopy(self.spielfeld))
                Spielfeldliste.append(deepcopy(self.get_turned_spielfeld(self.spielfeld))) ###
            else:
                Spielfeldliste.append(deepcopy(self.spielfeld))
                #Spielfeldliste.append(deepcopy(self.get_turned_spielfeld(self.spielfeld)))
            reward_liste.append(reward)
            #reward_liste[-2] -= reward_liste[-1]
            self.spieler1          = not self.spieler1
        
        self.reset()
        return Spielfeldliste, reward_liste
    
    
    def play_vs_rand(self):                                  # überprüfen
        Spielfeldliste = [deepcopy(self.spielfeld)]
        reward_liste   = [0.0]
        while not(np.array_equal(self.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(self.spielfeld[6:12] ,[0,0,0,0,0,0]) or self.spielfeld[12]>36 or self.spielfeld[13]>36):
            feld                   = self.get_next_action(self.spielfeld)
            print(self.guess_Q(self.spielfeld))
            #input()
            self.spielfeld, reward = self.get_spielfeld_and_reward_after_action(self.spielfeld, feld)
            #print(self.spielfeld , feld)

            
            Spielfeldliste.append(deepcopy(self.spielfeld))
            reward_liste.append(reward)
            #reward_liste[-2] -= reward_liste[-1]
            if not(np.array_equal(self.spielfeld[0:6] ,[0,0,0,0,0,0]) or np.array_equal(self.spielfeld[6:12] ,[0,0,0,0,0,0]) or self.spielfeld[12]>36 or self.spielfeld[13]>36):
                self.spieler1          = not self.spieler1
                feld                   = self.randomfeld(self.get_turned_spielfeld(self.spielfeld))
                self.spielfeld, reward = self.get_spielfeld_and_reward_after_action(self.spielfeld, feld)
                Spielfeldliste.append(deepcopy(self.spielfeld))
                self.spieler1          = not self.spieler1
                #print(self.spielfeld, feld)
                #input()
        #input()
        self.reset()
        return Spielfeldliste, reward_liste
    
    def get_win_rate(self, iterations):
        win  = 0
        loss = 0
        draw = 0
        exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0
        for it in range(iterations):
            s, r = self.play_vs_rand()
            #print(s)
            #input()
            if s[-1][12] > 36:
                win += 1
            elif s[-1][12] < 36:
                loss += 1
            else:
                draw += 1
        
        win_p  =win/iterations
        loss_p = loss/iterations
        draw_p = draw/iterations
        self.exploration_rate = exploration_rate
        print("Win: ", win_p, "Loss: ", loss_p,"draw: ", draw_p)
        return (win_p, loss_p, draw_p)
    
    def create_training_data(self, spielfeld_liste, reward_liste):
        q_liste    = []
        spielfeld2 = []
        reward1    = 0
        reward2    = 0
        q2         = []
        
        for i, r in zip(spielfeld_liste, reward_liste):
            q1    = self.guess_Q(i)
            legal = self.check_action(i)
            x     = np.zeros((len(q1),1))
            np.put(x, legal, np.ones((len(legal),1)))
            q1 = np.multiply(q1, x)
            
            
            for j in legal:
                spielfeld2, reward1    = self.get_spielfeld_and_reward_after_action(i, j) # hier kann man noch probieren
                self.spieler1          = False
                if np.array_equal(self.get_turned_spielfeld(spielfeld2)[0:6],[0,0,0,0,0,0]):
                    q1[j] += 1
                    self.spieler1          = True
                else:
                    q2                     = self.greedy_action(self.get_turned_spielfeld(spielfeld2)) # randomfeld
                
                    spielfeld2, reward2    = self.get_spielfeld_and_reward_after_action(spielfeld2, q2)
                    self.spieler1          = True
                
                    
                    q2                     = self.guess_Q(spielfeld2)
                    legal = self.check_action(spielfeld2)
                    x     = np.zeros((len(q2),1))
                    np.put(x, legal, np.ones((len(legal),1)))
                    q2 = np.multiply(q2, x)
                    reward1 = reward1 -0.5*reward2
                    q1[j]                  = (1-self.a)*q1[j] + self.a*(reward1 + self.discount * max(q2))
            
            q_liste.append(q1)
        return [(np.reshape(s,(14,1)),q) for s,q in zip(spielfeld_liste, q_liste)]

    
    
    def train_net(self, iterations, mini_batch_length, eta, epochs = 10):
        for i in (range(iterations)):
            spielfeld_liste, reward_liste = self.play()
            training_data = self.create_training_data(spielfeld_liste, reward_liste)
            #print(training_data)
            #input()
            if mini_batch_length > len(training_data):
                self.net.stochastic_update(training_data, len(training_data), eta, epochs)
            else:
                self.net.stochastic_update(training_data, mini_batch_length, eta, epochs)
        # evtl. speichern wir jedes mal / ab und zu die Gewichte und Bias (net.save_network.to_files(name))
        self.net.save_network_to_files(self.name)
    
    
    #################################################################################################################
    
    def guess_Q2(self, spielfeld):       # Uebergebe nur spielfeld bis feld 11?
        # guess the q-value of the current spielfeld
        tmp_spielfeld = deepcopy(spielfeld)
        
        gQ    = self.net2.feedforward(np.reshape(tmp_spielfeld,(14,1)))
        
        return gQ
    
    def create_training_data_dq(self, spielfeld_liste, reward_liste):
        q_liste    = []
        q_liste2   = []
        spielfeld2 = []
        s          = []
        reward1    = 0
        reward2    = 0
        
        for i, r in zip(spielfeld_liste[:-1], reward_liste[:-1]):
            legal = self.check_action(i)
            x     = np.zeros((6,1))
            np.put(x, legal, np.ones((len(legal),1)))
            
            q1 = self.guess_Q(i)
            q2 = self.guess_Q(i)
            q1 = np.multiply(q1,x)
            q2 = np.multiply(q2,x)
            
            
            action = self.get_next_action(i)
            
            spielfeld2, reward1 = self.get_spielfeld_and_reward_after_action(i, action)
            self.spieler1 = False
            spielfeld2, reward2 = self.get_spielfeld_and_reward_after_action(spielfeld2, self.greedy_action(self.get_turned_spielfeld(spielfeld2))) # randomfeld oder max reward
            self.spieler1 = True
            
            
            x2     = np.zeros((6,1))
            legal2 = self.check_action(spielfeld2)
            np.put(x2, legal, np.ones((len(legal2),1)))
            
            Q2 = np.multiply(self.guess_Q2(spielfeld2),x2)
            Q  = np.multiply(self.guess_Q(spielfeld2),x2)
            
            if self.net.act_func is Network.HyperbolicTangent:
                x2 = x2 - np.ones((len(Q),1))
                Q2 += x2
                Q  += x2
            
            q1[action][0] = (1-self.a)*q1[action][0]+ self.a*(reward1 + self.discount*Q2[np.argmax(Q)])
            
            q2[action][0] = (1-self.a)*q2[action][0]+ self.a*(reward1 + self.discount*Q[np.argmax(Q2)])
            
            s.append(np.reshape(i, (14,1)))
            q1 = np.multiply(q1,x)
            q2 = np.multiply(q2,x)
            if self.net.act_func is Network.HyperbolicTangent:
                x = x - np.ones((len(Q),1))
                q1 = q1 + x
                q2 = q2 + x
            q_liste.append(q1)
            q_liste2.append(q2)
            
        return [s, q_liste, q_liste2]
    
    def train_dq(self, iterations, mini_batch_length,eta,epochs = 10):
        for i in (range(iterations)):
            spielfeld_liste, reward_liste = self.play()
            training_data = self.create_training_data_dq(spielfeld_liste, reward_liste)
            if np.random.choice([0,1]) is 1:
                t = [(x,y) for x,y in zip(training_data[0],training_data[1])]
                if mini_batch_length > len(training_data):
                    self.net.stochastic_update(t, len(training_data), eta, epochs)
                else:
                    self.net.stochastic_update(t, mini_batch_length, eta, epochs)
            else:
                t  = [(x,y) for x,y in zip(training_data[0],training_data[2])]
                if mini_batch_length > len(training_data):
                    self.net2.stochastic_update(t, len(training_data), eta, epochs)
                else:
                    self.net2.stochastic_update(t, mini_batch_length, eta, epochs)
        # evtl. speichern wir jedes mal / ab und zu die Gewichte und Bias (net.save_network.to_files(name))
        self.net.save_network_to_files(self.name)
        self.net2.save_network_to_files(self.name2)
    
    def print_spielfeld(self):
        np.concatenate((self.spielfeld[13],self.spielfeld[6:11]), axis=None)
        np.concatenate((self.spielfeld[12],self.spielfeld[0:5]) , axis=None)
        print("\n{} | {}\n{} | {}\n".format(self.spielfeld[13],self.spielfeld[6:12], self.spielfeld[12], self.spielfeld[0:6]))
