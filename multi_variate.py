import numpy as np
from theano import *
import theano.tensor as T
import pandas as pd
import random

print("Loading dataset...")
df = pd.read_csv("data/floor_cost.csv", header=0)
data = df.as_matrix(columns=df.columns[:])

def show_data():
	print data[0:10]

z = T.dscalar('z')
x = T.dscalar('x')
w = theano.shared(np.array(0.), 'w')
target = T.dscalar('target')

## Equation 1
# model 1
y = z / x
f1 = theano.function([z, x], y)

## Equation 2
# model2
z = x * y
f2 = theano.function([x, y], z)

# def list1(): 
# 	list_x = [np.zeros(shape=(50,1))]
# 	for i in range(50):
# 		list_x[i] = data[i][0]
# 	return list_x

def list1():
	list_x = []
	for i in range(50):
		list_x.insert(i, data[i][0])
	return list_x

def list2(): #this is the list containing cost/tile type(eg, 124, 149, 521 etc)
	list_y = []
	for i in range(50):
		list_y.insert(i, data[i][1])
	return list_y

def list3(): #list containing requirements(eg, 391399, 171959,376434 etc)
	list_z = []
	for i in range(50):
		list_z.insert(i, data[i][2])
	return list_z

x1 = list1()
y1 = list2()
z1 = list3()

v2 = []
v1 = f2(x1[0], y1[0])
print "{}{}".format("Original Value: ", v1)

#shuffle the list
c = zip(y1, z1)
random.shuffle(c)
y1, z1 = zip(*c)

for x in range(50):
	v2.insert(x, f2(x1[0], y1[x]))
	#print "{}{}{}".format(x, ". ", v2)

# def match(x_val, orig):	
# 	for u in range(50):
# 		if v2[u]==orig:
# 			print "{}{}".format("Match found at y-index: ", u)
# 	return v2[u]

		
# chkmatch = match(x1[0], v1)
#print chkmatch



def diff(value1, value2):
	diff = value1 - value2
	return diff

def abs_rel_err(a, b, eps=1.0e-10):
	return abs(a - b) / (abs(a) + abs(b) + eps)

#ans = abs_rel_err(v1, v2)
#print ans
#ans = diff(v1, v2)

# for s in range(50):
# 	res = diff(v1, v2[s])
# 	print "{}{}{}".format(s,". ", res)
# 	if res==0.0:
# 		print ("Match found!!!")
ans = []
for m in range(50):
	ans.insert(m, abs_rel_err(v1, v2[m]))
for el in ans:
	print el

x = min(float(s) for s in ans)
print "{}{}".format("The minimum error value/matching value is: ", x)