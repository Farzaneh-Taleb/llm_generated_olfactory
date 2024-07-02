from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV, MultiTaskLassoCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA




def pipeline_regression(X_train,y_train,X_test,regression_method,seed,n_components=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if n_components is not None:
        pca = PCA(n_components=n_components)
        X_train=pca.fit_transform(X_train)
        X_test=pca.transform(X_test)
    linreg =regression_method(X_train,y_train,seed)
    return linreg,X_test

def custom_logistic_regression(X,y,seed):
    clf = LogisticRegressionCV(cv=5, random_state=seed,max_iter=200,scoring='roc_auc')
    multi_clf =  OneVsRestClassifier(clf,n_jobs=-1)
    estimator= multi_clf.fit(X,y)
    return estimator


def custom_linear_regression(X, y, seed):
    # print(y.shape)
    if len(y.shape) > 1:
        linreg = MultiTaskLassoCV(max_iter=1000, n_alphas=200, random_state=seed, n_jobs=-1)
    else:
        linreg = LassoCV(max_iter=1000, n_alphas=200, random_state=seed, n_jobs=-1)

    estimator = linreg.fit(X, y)
    return estimator