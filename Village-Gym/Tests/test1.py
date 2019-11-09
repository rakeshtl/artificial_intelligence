import random

def sellCorn(week):
    values = []
    prob = []
    if week == 9:
        values = [50,100]
        prob = [0.5,0.5]
        return random.choices(values, prob)[0]
    elif week == 10:
        values = [60,100]
        prob = [0.4,0.6]
        return random.choices(values, prob)
    elif week == 11:
        values = [70,100]
        prob = [0.3,0.7]
        return random.choices(values, prob)
    elif week == 12:
        values = [80,100]
        prob = [0.2,0.8]
        return random.choices(values, prob)
    elif week == 13:
        values = [90,100]
        prob = [0.05,0.95]
        return random.choices(values, prob)
    return 100
    
print(sellCorn(9))