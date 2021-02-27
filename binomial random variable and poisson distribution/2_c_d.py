import math
import numpy as np
import matplotlib.pyplot as plt

def factorial(k):
  res = 1
  for i in range(2,k+1):
    res *= i
  return res

def poisson_pmf(lamda,k):
  return math.exp(-lamda)*math.pow(lamda,k)/factorial(k)

# 2_c
n = 1000
size = 10000
res = np.random.binomial(p=0.01, n=n, size = size)
plt.hist(res)
plt.xlabel('k')
plt.ylabel('frequency')
plt.title('2-c')
plt.show()
plt.clf()

# 2_d
x_range = [i for i in range(30)]
pmf_list = []
for x in x_range:
  pmf_list.append(poisson_pmf(10,x))
plt.plot(x_range,pmf_list)
plt.title('2-d. pmf')
plt.show()