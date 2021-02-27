import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../iris/iris.data', delimiter=',', names=['sepal_length_in_cm','sepal_width_in_cm','petal_length_in_cm','petal_width_in_cm','class'])

sns_plot = sns.scatterplot(data=data, x = 'sepal_length_in_cm', y = 'sepal_width_in_cm', hue = 'class')
plt.plot([4.45,6.3],[2.0,4.5])
plt.plot([6.25,6.25],[2.0,4.5])
plt.legend()
fig = sns_plot.get_figure()
fig.savefig('../plots/scatter_0.png')
plt.clf()

sns_plot = sns.scatterplot(data=data, x = 'petal_length_in_cm', y = 'petal_width_in_cm', hue = 'class')
plt.plot([1,3.5],[1.8,0.0])
plt.plot([2.25,6.25],[3.6,0.5])
fig = sns_plot.get_figure()
fig.savefig('../plots/scatter_1.png')
plt.clf()

sns_plot = sns.scatterplot(data=data, x = 'sepal_length_in_cm', y = 'petal_length_in_cm', hue = 'class')
plt.plot([4.2,8],[2.3,2.3])
plt.plot([4.2,8],[4.2,5.3])
fig = sns_plot.get_figure()
fig.savefig('../plots/scatter_2.png')
plt.clf()

sns_plot = sns.scatterplot(data=data, x = 'sepal_width_in_cm', y = 'petal_width_in_cm', hue = 'class')
plt.plot([1.8,4.5],[0.8,0.8])
plt.plot([1.8,4.5],[1.65,1.65])
fig = sns_plot.get_figure()
fig.savefig('../plots/scatter_3.png')
plt.clf()

sns_plot = sns.scatterplot(data=data, x = 'sepal_width_in_cm', y = 'petal_length_in_cm', hue = 'class')
plt.plot([1.8,4.5],[2.2,2.2])
plt.plot([1.8,4.5],[4.7,4.7])
fig = sns_plot.get_figure()
fig.savefig('../plots/scatter_4.png')
plt.clf()


sns_plot = sns.scatterplot(data=data, x = 'petal_width_in_cm', y = 'sepal_length_in_cm', hue = 'class')
plt.plot([0.8,0.8],[4.3,8.0])
plt.plot([1.65,1.65],[4.3,8.0])
fig = sns_plot.get_figure()
fig.savefig('../plots/scatter_5.png')
plt.clf()
