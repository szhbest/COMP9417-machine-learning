import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

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
    select_c = [1e-3, 1e-2, 1e-1, 1, 1e1, 3e1, 1e2, 3e2, 1e3]
    for c in select_c:
        test_accuracy = 0
        train_accuracy = 0
        for k in k_dict:
            X_train, X_test, y_train, y_test = k_dict[k]

            # For each training set, do Standardization, and apply its rule to the corresponding test set
            std_scale = StandardScaler().fit(X_train)
            X_train = std_scale.transform(X_train)
            X_test = std_scale.transform(X_test)

            # LogisticRegression model
            clf = LogisticRegression(penalty='l2', max_iter=100, random_state=0, solver='liblinear', C=c).fit(X_train, y_train)

            test_accuracy += compute_acc(clf, X_test, y_test)
            train_accuracy += compute_acc(clf, X_train, y_train)
        print(f"{c}: train accuracy: {train_accuracy / length}, test accuracy: {test_accuracy / length}")


if __name__ == '__main__':
    input_X, label_y = preprocessing()
    dict = k_fold(input_X, label_y, 5)
    use_cv(dict)