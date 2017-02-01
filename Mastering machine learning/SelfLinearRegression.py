import numpy as np


def checkData(X, Y):
    return len(X) == len(Y) and len(X) > 0 and len(X[0]) > 0


def updateWeights(weights, alpha, train_x, train_y):
    n_features = train_x.shape[1]
    n_size = train_x.shape[0]
    new_weights=np.array(weights)
    a = alpha
    for i in range(n_features+1):
        sum=0
        for j in range(n_size):
            if i == 0:
                sum += ((train_x[j].dot((weights[0][1:].T))) - train_y[j])
            else:
                sum += (((train_x[j].dot((weights[0][1:].T))) - train_y[j]) * train_x[j][i-1])
        new_weights[0][i]-=(a*sum)/n_size
    return new_weights



def featureScale(X):
    sumVector=np.sum(X,axis=0)
    stdVector=np.std(X,axis=0)

    for i in range(X.shape[1]):
        X[:,[i]]=(X[:,[i]]-(sumVector[i]/X.shape[0]))/stdVector[i]

    return X


def fit(X, Y, n_iter=10, alpha=0.1):
    if checkData(X, Y):
        train_x = np.array(X)
        train_y = np.array(Y)
        train_x=featureScale(train_x)
        weights = np.zeros((1, len(X[0]) + 1))
        for i in range(n_iter):
            weights = updateWeights(weights, alpha, train_x, train_y)
        return weights

    return -1


x = [[1], [2], [3]]
y = [1,2,3]

print(fit(x, y))

# todo- feature scale
