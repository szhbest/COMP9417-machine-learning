import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

#data pre-processing
def preprocessing():
    # get label y
    # y = pd.read_csv('F/FS_post.csv', header=0)
    # y = pd.read_csv('P_po/panas_negative_post.csv', header=0)
    y = pd.read_csv('P_po/panas_positive_post.csv', header=0)
    y = np.array(y)[:, 1:]
    y = y.reshape(-1)
    # get input X
    # X = pd.read_csv('F/input_top5.csv', header=0)
    # X = pd.read_csv('P_neg/input_top5.csv', header=0)
    X = pd.read_csv('P_po/input_top5.csv', header=0)
    X = np.array(X)[:, 1:].astype(np.float64)
    # use  Pearson's Correlation to select useful features
    X = SelectKBest(lambda A, B: tuple(map(tuple, np.array(list(map(lambda a: pearsonr(a, B), A.T))).T)),
                    k=10).fit_transform(X, y)
    return X, y

#k-fold cross validation
def k_fold(X, y, n):
    kf = KFold(n_splits=n, shuffle=True, random_state=0)
    seq = 1
    k_fold_dict = {}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        k_fold_dict[seq] = [X_train, X_test, y_train, y_test]
        seq += 1
    return k_fold_dict

#MSE computation
def compute_mse(model, test_x, test_y):
    prd_y = model.predict(test_x)
    length = len(prd_y)
    mse = 0
    for i in range(length):
        mse += (test_y[i] - prd_y[i])**2
    return mse / length

#The mean training MSE and test MSE of the model under different parameters
def use_cv(k_dict):
    length = len(k_dict)
    alpha_set = [0.01, 0.1, 1, 10, 100, 1000]
    hidden_size = [10, 50, 100, 200, 300]
    dd = {}
    # based on a particular alpha, to choose a optimal hidden_size number
    for h in hidden_size:
        test_mse = 0
        train_mse = 0
        for k in k_dict:
            X_train, X_test, y_train, y_test = k_dict[k]

            # For each training set, do Standardization, and apply its rule to the corresponding test set
            std_scale = StandardScaler().fit(X_train)
            X_train = std_scale.transform(X_train)
            X_test = std_scale.transform(X_test)

            #ann model
            clf = MLPRegressor(random_state=0, hidden_layer_sizes=(h, ), activation='logistic', solver='adam',
                               alpha=10, max_iter=10000).fit(X_train, y_train)

            test_mse += compute_mse(clf, X_test, y_test)
            train_mse += compute_mse(clf, X_train, y_train)
        print(f'train mse: {train_mse / length}   test mse: {test_mse / length}')
        dd[h] = [train_mse / length, test_mse / length]

    # to plot the "Bias-Variance Tradeoff"
    y_ = [i for i in dd]
    mse_train = [dd[i][0] for i in dd]
    mse_test = [dd[i][1] for i in dd]
    plt.plot(y_, mse_train, 'r', label='train mse')
    plt.plot(y_, mse_test, 'b', label='test mse')
    plt.legend()
    plt.xlabel('The number of hidden units')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    input_X, label_y = preprocessing()
    dict = k_fold(input_X, label_y, 5)
    use_cv(dict)


