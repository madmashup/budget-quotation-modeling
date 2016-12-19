import numpy as np
from theano import *
import theano.tensor as T
import pandas as pd

print("Loading dataset...")
df = pd.read_csv("data/floor_var2.csv", header=0)
data = df.as_matrix(columns=df.columns[:])

x = T.dscalar('x')
y = T.dscalar('y')
w = theano.shared(np.asarray(0.), 'w')
c = theano.shared(np.asarray(0.), 'c')
target = T.dscalar('target')

w = x / y
f = theano.function([x, y], w)

def rmse():
	cost = T.mean(T.sqr(c - target))
	return cost

def grad_calculation(cost_val):
	grad = T.grad(cost=cost_val, wrt=c)
	w_updated = c - (0.1 * grad)
	update = [[c, w_updated]]
	return update

def cost_iterate(actual_out, target_value, itr):
	cos = rmse()
	b = grad_calculation(cos)
	func = theano.function(inputs=[x, target], outputs=cos, allow_input_downcast=True, updates=b, on_unused_input='ignore')
	for i in xrange(itr):
		res = func(actual_out, target_value)
		print res

cost_iterate(data[0][2], 124, 30)

