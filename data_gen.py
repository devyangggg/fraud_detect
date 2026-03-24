import numpy as np
import sys

def generate_snapshot(mid_price):
    rand = np.random.default_rng()
    res_vec = []

    for i in range(0,3):
        c_speed = rand.uniform(0, 0.3)
        price = rand.integers(0,mid_price)
        quantity = rand.integers(0,mid_price*2)
        res_vec.append([price,quantity,c_speed])

    for i in range(3,6):
        c_speed = rand.uniform(0, 0.3)
        price = rand.integers(mid_price,mid_price*2)
        quantity = rand.integers(0,mid_price*2)
        res_vec.append([-price,quantity,c_speed])

    res_vec = np.vstack(res_vec)

    return res_vec

generate_snapshot(1000)


def generate_window():

    res = []
    for i in range(0,100):
        vec = generate_snapshot(1000)
        res.append(vec)

    res = np.stack(res)
    return res

        

def generate_spoof_snapshot(mid_price):
    rand = np.random.default_rng()
    vec = []
    for i in range(0,3):
        c_speed = rand.uniform(0.7, 1)
        price = rand.integers(0,mid_price)
        quantity = rand.integers(mid_price*1.5,mid_price*2)
        vec.append([price,quantity,c_speed])

    for i in range(3,6):
        c_speed = rand.uniform(0, 0.3)
        price = rand.integers(mid_price,mid_price*2)
        quantity = rand.integers(0,mid_price*2)
        vec.append([-price,quantity,c_speed])

    vec = np.vstack(vec)
    return vec


def generate_spoof_window(mid_price):

    res = []
    for i in range(0,100):
        
        if(i > 40 and i<60):
            vec = generate_spoof_snapshot(mid_price)
            res.append(vec)
            
        else:
            vec = generate_snapshot(mid_price)
            res.append(vec)

    res = np.stack(res)
            
    return res

generate_spoof_window(1000)

def generate_data(mid,n):
    vec = []
    half = n//2

    for i in range(0,half):
        vec.append(generate_window())

    for i in range(0,half):
        vec.append(generate_spoof_window(1000))

    vec = np.stack(vec)

    y = np.array([0]*(n//2) + [1]*(n//2))

    return vec, y
    

generate_data(1000, 100)