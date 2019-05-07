import numpy as np
import matplotlib.pyplot as plt
w = np.zeros((2,1))
q = np.ones((2,1))
step = 1
def f(w):
    return (np.sqrt((w-q).transpose().dot((w-q))))
def derivate(w):
    return (w-q)/ f(w)
res = []
w_list = []
res.append(f(w)[0])
for i in range(49):
    diff = derivate(w)
    w = w - step * diff
    w_list.append(w)
    res.append(f(w)[0])
    
    
    
plt.figure()
plt.plot(res)
#%%
res = []

w = np.zeros((2,1))
q = np.ones((2,1))
res.append(f(w)[0])
for i in range(1,50):
    diff = derivate(w)
    step = 1/np.sqrt(i)
    w = w - step * diff
    res.append(f(w)[0])
    
plt.figure()
plt.plot(res)
