import numpy as np
import matplotlib.pyplot as plt

p_heads_list = []
for i in range(100):
  res = np.random.binomial(n=1, p=0.5, size=i+1)
  p_heads = np.count_nonzero(res == 1)/(i+1)
  p_heads_list.append(p_heads)

x_range = [i for i in range(1,101)]
plt.plot(x_range,p_heads_list)
plt.xlabel('the number of coin tosses')
plt.ylabel('the proportions of heads')
plt.grid(True)
plt.show()