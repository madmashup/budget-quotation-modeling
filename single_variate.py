import numpy as np
from theano import *
import theano.tensor as T
import pandas as pd

print("Loading dataset...")
df = pd.read_csv("data/floor.csv", header=0)
data = df.as_matrix(columns=df.columns[:])

# print("Your dataset schema looks like this!")
# print(data[1:10])

#input
# s = raw_input()
# numbers = map(int, s.split())
# print("Input your numbers...")
# a = [int(m) for m in raw_input().split()]
# b = [int(n) for n in raw_input().split()]
# c = [int(o) for o in raw_input().split()]

x = T.dscalar('x')
y = T.dscalar('y')
z = T.dscalar('z')
#we can take x as a shared variable too!!
#eg, x = theano.shared(np.asarray(1000.), 'x')

#model
y = z/x
f = theano.function([z, x], y)

def grade(ans):
    if 250<=ans<400:
        print 'B'
    elif ans>=400:
        print 'A'
    elif 100<=ans<250:
        print 'C'
    elif 50<=ans<100:
        print 'D'
    else:
        print 'Wrong Output'

def model():
    for i in range(len(data)):
        res = f(data[i][2], data[i][0])
        a = grade(res) 
    return a

b = model()
print b
