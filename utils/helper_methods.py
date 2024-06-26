from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA



def pipeline_regression(X_train,y_train,X_test,seed,n_components=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if n_components is not None:
        pca = PCA(n_components=n_components)
        X_train=pca.fit_transform(X_train)
        X_test=pca.transform(X_test)
    linreg =custom_logistic_regression(X_train,y_train,seed)
    return linreg,X_test

def custom_logistic_regression(X,y,seed):
    clf = LogisticRegressionCV(cv=5, random_state=seed,max_iter=200,scoring='roc_auc')
    multi_clf =  OneVsRestClassifier(clf,n_jobs=-1)
    estimator= multi_clf.fit(X,y)
    return estimator