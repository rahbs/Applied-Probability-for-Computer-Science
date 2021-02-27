import numpy as np
import matplotlib.pyplot as plt

# 2_a
samples = 1024
n = 10

k_list = []
for _ in range(samples):
  cnt = 0
  for _ in range(n):
    if np.random.randint(0,2) == 1:
      cnt = cnt+1
  k_list.append(cnt)
plt.hist(k_list)
plt.xlabel('k')
plt.ylabel('frequency')
plt.show()

# 2_b
# P(p_hat = 0.5)
print('P(p_hat = 0.5): ', k_list.count(5)/samples)