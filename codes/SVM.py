import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn import svm

#data pre-processing
def preprocessing():
    # get label y
    # y = pd.read_csv('F/FS_post.csv', header=0)
    # y = pd.read_csv('P_neg/panas_negative_post.csv', header=0)
    y = pd.read_csv('P_po/panas_positive_post.csv', header=0)
    y = np.array(y)[:, 1:]
    # use median value as the threshold
    thr = np.median(y)
    # divide y into two groups
    y = Binarizer(threshold=thr).fit_transform(y)
    y = y.reshape(-1)

    # get input X
    # X = pd.read_csv('F/input_top5.csv', header=0)
    # X = pd.read_csv('P_neg/input_top5.csv', header=0)
    X = pd.read_csv('P_po/input_top5.csv', header=0)
    X = np.array(X)[:, 1:].astype(np.float64)
    # use  Pearson's Correlation to select useful features
    X = SelectKBest(lambda A, B: tuple(map(tuple, np.array(list(map(lambda a: pearsonr(a, B), A.T))).T)), k=10).fit_transform(X, y)
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

#accuracy computation
def compute_acc(model, test_x, test_y):
    prd_y = model.predict(test_x)
    length = len(prd_y)
    error = 0
    for i in range(length):
        if prd_y[i] != test_y[i]:
            error += 1
    return 1 - error / length

#The mean training accuracy and test accuracy of the model under different parameters
def use_cv(k_dict):
    length = len(k_dict)
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for c in C:
        accuracy_test = 0
        accuracy_train = 0
        for k in k_dict:
            X_train, X_test, y_train, y_test = k_dict[k]

            # For each training set, do Standardization, and apply its rule to the corresponding test set
            standScale = StandardScaler().fit(X_train)
            X_train = standScale.transform(X_train)
            X_test = standScale.transform(X_test)

            # linearSVC model
            clf = svm.LinearSVC(penalty='l2',loss='squared_hinge',dual=True, C=c, max_iter=10000).fit(X_train, y_train)

            accuracy_test += compute_acc(clf, X_test, y_test)
            accuracy_train += compute_acc(clf, X_train, y_train)
        print(accuracy_test / length)
        print(accuracy_train / length)



if __name__ == '__main__':
    input_X, label_y = preprocessing()
    dict = k_fold(input_X, label_y, 5)
    use_cv(dict)



