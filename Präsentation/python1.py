if Spieler1gewonnen > 80:
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
        