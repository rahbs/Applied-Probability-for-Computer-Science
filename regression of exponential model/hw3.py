import pandas as pd
import numpy as np

def cal_linear_params (x, y):
    n = len(y)
    a_1 = (n * np.sum(y * x) - np.sum(y) * np.sum(x)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    a_0 = np.mean(y)-a_1 * np.mean(x)
    return a_0, a_1

def exp_func(x, a, b):
    return a * np.exp(b * x)

def linear_func(x, a, b):
    return a*x +b
data = pd.read_csv('OilProduction.txt',skiprows=11,sep = '\t')

# truncate data to obtain data before 1973
truncated_data=data[:21]

xdata = np.array(data['Year'])
ydata = np.array(data['Mbbl'])
truncated_xdata = np.array(truncated_data['Year'])
truncated_ydata = np.array(truncated_data['Mbbl'])

# y = ae^(bx)
# z = ln(y), a_0 = ln(a), a_1 = b
# z = a_0 + a_1x

# calculate parameters using data before 1973
z = np.log(truncated_ydata)
t_a_0, t_a_1 = cal_linear_params(truncated_xdata, z)
t_a = np.exp(t_a_0)
t_b = t_a_1
print('parameters using data before 1973: ')
print('a_0: ',round(t_a_0,4), '  a_1: ',round(t_a_1,4), ' (parameters for linear model)')
print('a: ',format(t_a,"10.2e"),'   b: ',format(t_b,"10.2e"),' (parameters for exponential model)\n')

# calculate paramters using total data
z = np.log(ydata)
a_0, a_1 = cal_linear_params (xdata, z)
a = np.exp(a_0)
b = a_1
print('parameters using total data: ')
print('a_0: ',round(a_0,4), '  a_1: ',round(a_1,4), ' (parameters for linear model)')
print('a: ',format(a,"10.2e"),'   b: ',format(b,"10.2e"),' (parameters for exponential model)\n')

# transform Mbbl data
transformed_ydata = np.log(ydata)

print("natural-log-transformed data: ")
for x, i in zip(xdata,transformed_ydata):
    print(x,'\t',i)