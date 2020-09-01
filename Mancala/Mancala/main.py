import Network as net
import mancala as man
import numpy as np
import os

a = man.Mancala(exploration_rate = 1.0)

a.name = "Net"
a.name2 = "Net2"
a.net.generate_random_network([14,14,14,14,14,6])
a.net2.generate_random_network([14,14,14,14,14,6])


rate = []

size = 1000
for i in range(size):
    print("Training")
    a.train_dq(100, 10, 0.2, 10)
    a.exploration_rate -= 1/(size + size /10)
    rate.append(a.get_win_rate(1000))

np.savetxt("rate.csv",rate,delimiter=',')
