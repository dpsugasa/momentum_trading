from cvxpy import *
import numpy as np

np.random.seed(1)
n = 10
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)

betas = [np.random.uniform(-1,1) for _ in range(10)]

w = Variable(n)

risk = quad_form(w, Sigma)
constraints = [sum_entries(w) == 1, w >= 0, w.T*beta == 0]
prob = Problem(Minimize(risk), constraints)

#for i in range(100):
prob.solve()

print('Weights :', w.value)

w.T*beta == 0