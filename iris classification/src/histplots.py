import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../iris/iris.data', delimiter=',', names=['sepal_length_in_cm','sepal_width_in_cm','petal_length_in_cm','petal_width_in_cm','class'])

sns_plot = sns.histplot(data=data,x='sepal_length_in_cm', hue = 'class', bins = 30)
plt.plot([5.5,5.5],[12,0])
plt.plot([6.2,6.2],[12,0])
fig = sns_plot.get_figure()
fig.savefig('../plots/hist_sepal_length.png')
plt.clf()

sns_plot = sns.histplot(data=data,x='sepal_width_in_cm', hue = 'class', bins = 30)
plt.plot([3.36,3.36],[12,0])
plt.plot([2.8,2.8],[12,0])
fig = sns_plot.get_figure()
fig.savefig('../plots/hist_sepal_width.png')
plt.clf()

sns_plot = sns.histplot(data=data,x='petal_length_in_cm', hue = 'class', bins = 30)
plt.plot([2.5,2.5],[25,0])
plt.plot([4.75,4.75],[25,0])
fig = sns_plot.get_figure()
fig.savefig('../plots/hist_petal_length.png')
plt.clf()

sns_plot = sns.histplot(data=data,x='petal_width_in_cm', hue = 'class', bins = 30)
plt.plot([0.8,0.8],[25,0])
plt.plot([1.75,1.75],[25,0])
fig = sns_plot.get_figure()
fig.savefig('../plots/hist_petal_width.png')
plt.clf()
