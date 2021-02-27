import pandas as pd
import math
import numpy as np

from functions import multivariate_gaussian_bayes_classifier, get_covariance_matrix, calculate_mean, calculate_std, calculate_precision, calculate_recall, calculate_avg

data = pd.read_csv('../iris/iris.data', delimiter=',', names=['sepal_length_in_cm','sepal_width_in_cm','petal_length_in_cm','petal_width_in_cm','class'])

# preprocessing 
# Iris-setosa : 0 / Iris-versicolor: 1 / Iris-virginica : 2
data.set_value(data['class']=='Iris-setosa','class',0)
data.set_value(data['class']=='Iris-versicolor','class',1)
data.set_value(data['class']=='Iris-virginica','class',2)

# 5-fold cross validation
k = 5
n_class = 3
setosa_data = data[data['class']== 0]
versicolor_data = data[data['class']== 1]
virginica_data = data[data['class']== 2]
sep_data_by_class = [setosa_data,versicolor_data, virginica_data]

precision_list = []
recall_list = []

for i in range(k):
    print((i+1),'- th fold')
    # get train and validation data for k-fold cross validation
    # 각 fold의 train, valid data에 클래스 비율을 같게 해주었음
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    for sep_data in sep_data_by_class:
        tmp = sep_data
        tmp = tmp.drop(tmp.index[i*10:(i+1)*10])
        train_data = pd.concat([train_data,tmp])
        val_data = pd.concat([val_data, sep_data[i*10:(i+1)*10]])
       
    # get mean_list / std_list of train_data for each class and feature
    # 각 fold마다 해당 training data에 대한 mean / std를 구함
    n_cols = len(train_data.columns)
    mean_vector_list = []
    covar_matrix_list = []
    for class_ in range (n_class):
        x_ = train_data[train_data['class'] == class_]
        x_matrix = np.array( x_.iloc[:,:-1].values)
        covar_matrix, mean_vector = get_covariance_matrix(x_matrix)
        covar_matrix_list.append(covar_matrix)
        mean_vector_list.append(mean_vector)
#     print('covar_matrix: \n',covar_matrix)
#     print('mean_vector_list: ',mean_vector_list)
    # inference  
    val_class_list = val_data['class'].tolist()
    val_class_hat_list = []
    for idx, row in val_data.iterrows():
        val_x = [row.sepal_length_in_cm, row.sepal_width_in_cm, row.petal_length_in_cm,row.petal_width_in_cm]
        # gaussian classifer
#         print('val_x: ',val_x)
        val_class_hat = multivariate_gaussian_bayes_classifier(val_x, mean_vector_list, covar_matrix_list)
        val_class_hat_list.append(val_class_hat)
        print('ground truth class: ',val_data['class'][idx], '  predicted class: ',val_class_hat)

    # evaluate classifer
    # precision of each feature
    precision = calculate_precision(val_class_list,val_class_hat_list,n_class)
    # recall of each feature
    recall = calculate_recall(val_class_list,val_class_hat_list, n_class) 
    precision_list.append(precision)
    recall_list.append(recall)
    print('precision: ', precision,'\nrecall: ', recall)
    print("*"*50)

avg_precision = calculate_avg(precision_list)
avg_recall = calculate_avg(recall_list)
print('- average precision and recall of 5-folds for each class:')
print('avg_precision: ', avg_precision,'\navg_recall: ', avg_recall)