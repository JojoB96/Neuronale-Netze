#Mancala

#Files
Readme.md
Mancala/__init__.py
Mancala/class.py

Struktur Netz:

#def __init__
Netz

#def spielzug1/2
Input: Spielzug
Output: neues Spielfeld

#def guess_Q
Input: Spielfeld
schätzt den Q-Wert
Output: activation

#def greedy_action
Input: Spielfeld
Output: Feld mit größter activation

#def get_next_action
Input: Spielfeld
Output: Feld mit größter activation oder Randomfeld

#def create_training_data
Input: Spielfeldliste
Output: Vektor aus Spielfeldliste und Q-Values

# def train_net(self, iterations):
Input: maximale Anzahl an Iterationen
erstelle Spielfeldliste indem ein Spiel durchgeführt wird
create_training_data
train (Backpropagation, update_weights)
"Output":speichere trainiertes Netz (jede x-te Iteration)



################################################################

#zu Spielregeln einbauen
Vorschlag:
#def take_action
Input: Feld
mache von Feld aus nach Regeln einen Zug
bestimme Reward
Output: Spielfeld
