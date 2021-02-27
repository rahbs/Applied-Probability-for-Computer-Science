import math
import numpy as np

def calculate_mean(x):
    length = len(x)
    sum = 0
    for elem in x:
        sum += elem
    return sum/length

def calculate_std(x,mean):
    length = len(x)
    sum = 0
    for elem in x:
        sum += (elem-mean)**2
    return math.sqrt(sum/(length-1))

def find_max_idx(n_list):
    num = n_list[0]
    max_idx = 0
    for idx, n in enumerate(n_list):
        if num < n:
            num = n
            max_idx = idx
    return max_idx

# interval로 data를 나누어, 그 구간에 대한 likelihood를 구한다.
def pmf(x,data,interval):
    data = sorted(data)
    length = len(data)
    min_data = data[0]
    max_data = data[-1]
    bins = int((max_data-min_data)/interval)+1

    # likelihood가 0일 경우, 0 대신 epslion 값으로 설정해준다.
    # feature 중에 하나라도 0이 나오면, 곱해서 전체가 0이 되는 것을 방지하기 위함
    epsilon = 0.000001
    
    # x가 속한 구간을 찾는다
    for i in range(bins):
        cnt = 0
        separator = min_data + interval*i
        #print('separator: ',separator)
        if x < separator:
            if i == 0:
                return epsilon
            # 해당 구간의 likelihood를 구한다.
            for d in data:
                if separator-interval <= d and d < separator:
                    cnt += 1
                elif d >= separator:
                    break
            break
    likelihood = cnt/length
    if likelihood == 0:
        likelihood = epsilon
    return likelihood

# bayesian classifier
# x_list shape: (n_cols)
# output: predicted class index
def bayesian_classifier(x_list,data,n_class,interval):
    posterior_list = []
    # set prior to 1/3 
    prior = 1/3
    # 각 클래스별로 posterior를 구한다
    for class_idx in range(n_class):
        data_per_class = data[data['class']==class_idx]
        likelihood = 1
        for feature_idx, x in enumerate(x_list):
            data_per_class_feature = data_per_class.iloc[:,feature_idx].tolist()
            pmf_ = pmf(x,data_per_class_feature,interval = 0.2)
            likelihood *= pmf_
        posterior = prior * likelihood
        posterior_list.append(posterior)
    # find class which has max posterior 
    predicted_class = find_max_idx(posterior_list)
    return predicted_class

# gaussian function
def gaussian(mu, sigma, x):
    return math.exp(-((x-mu)**2)/(2*(sigma**2)))/math.sqrt(2 * math.pi*(sigma**2))

# gaussian bayes classifier
# x_list shape: (n_cols)
# mean_list, std_list shape: (n_class, n_cols)
def gaussian_bayes_classifier(x_list, mean_list, std_list):
    posterior_list = []
    # set prior to 1/3 
    prior = 1/3
    for means, stds in zip(mean_list, std_list):
        likelihood = 1
        for x, mean, std in zip(x_list,means, stds):
            likelihood *= gaussian(mean, std, x)
        posterior = prior * likelihood
        posterior_list.append(posterior)
    # find class which has max posterior 
    predicted_class = find_max_idx(posterior_list)
    return predicted_class

def get_covariance_matrix(x_matrix):
    n_samples = x_matrix.shape[0]
    n_features = x_matrix.shape[1]
    x_matrix_t = x_matrix.T
    mean_vector = np.empty([n_features,1])
    for i,x in enumerate(x_matrix_t) :
        mean_vector[i][0] = calculate_mean(x)
    x_matrix_t = x_matrix_t-mean_vector
    x_matrix = x_matrix - mean_vector.T
    
    return np.dot(x_matrix_t,x_matrix)/(n_samples-1), mean_vector.squeeze()

def multivariate_gaussian(x_vector,mean_vector,covar_matrix):
    inv_covar_matrix = np.linalg.inv(covar_matrix)
    X = x_vector-mean_vector
    res = math.exp((-1/2)*np.dot(np.dot(X,inv_covar_matrix),X))/2*(math.pi**2)*math.sqrt(np.linalg.det(covar_matrix))
    return res

# multivariate gaussian bayes classifier
# x_vector shape: (n_cols)
# mean_vector_list shape: (n_class, n_cols)
# covar_matrix_list shape : (n_class, n_cols, n_cols)
def multivariate_gaussian_bayes_classifier(x_vector, mean_vector_list, covar_matrix_list):
    posterior_list = []
    # set prior to 1/3 
    prior = 1/3
    for mean_vector, covar_matrix in zip(mean_vector_list, covar_matrix_list):
        likelihood = multivariate_gaussian(x_vector, mean_vector, covar_matrix)
        posterior = prior * likelihood
        posterior_list.append(posterior)
    # find class which has max posterior 
    predicted_class = find_max_idx(posterior_list)
    return predicted_class

# return precision_list for each class
def calculate_precision(c,c_hat,n_class):
    precision_list = []
    for i in range(n_class):
        # a = TP
        a = 0.0
        # b = TP + FP
        b = 0.0
        for e, e_hat in zip(c, c_hat):
            if e_hat == i:
                b += 1
                if e_hat == e:
                    a += 1
        # precision: TP/(TP +FP)
        precision_list.append(a/b)
    return precision_list
                       
# return recall_list for each class
def calculate_recall(c,c_hat,n_class):
    recall_list = []
    for i in range(n_class):
        # a = TP
        a = 0.0
        # b = TP + FN
        b = 0.0
        for e, e_hat in zip(c, c_hat):
            if e == i:
                b += 1
                if e_hat == e:
                    a += 1
        # recall : TP/(TP +FN)
        recall_list.append(a/b)
    return recall_list

# calculate average from 2D list (axis = 0)
def calculate_avg(x):
    n_row = len(x) 
    n_col = len(x[0])
    avg_list = []
    for idx in range(n_col):
        sum = 0.0
        for r in x:
            sum += r[idx]
        avg = sum/n_row
        avg_list.append(avg)
    return avg_list