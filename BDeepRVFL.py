import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.io import loadmat
import pymc3 as pm

n_node = 10 # num of nodes in hidden layer
n_layer = 5 # num of hidden layers
w_range = [-1, 1] # range of random weights
b_range = [0, 1] # range of random biases

class BayesianDeepRVFL:
    """ Bayesian Deep RVFL Classifier """

    def __init__(self, n_node, w_range, b_range, n_layer, alpha_1=10**(-5), alpha_2=10**(-5), alpha_3=10**(-5), alpha_4=10**(-5), n_iter=1000, tol=1.0e-3, activation='relu', same_feature=False):
        self.n_node = n_node
        self.w_range = w_range
        self.b_range = b_range
        self.n_layer = n_layer
        
        self.alpha_1 = alpha_1 # Gamma distribution parameters
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.n_iter = n_iter
        self.tol = tol

        self.weight = []
        self.bias = []
        self.beta = None
        self.prec = None
        self.var = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.covar = None
        self.mean = None
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature

    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1

        n_sample, n_feature = np.shape(data)
        d = self.standardize(data, 0) # Normalize
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
            self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, d], axis=1)

        # d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1) # concat column of 1s
        y = self.one_hot_encoding(label, n_class)
        dT_y = np.dot(d.T, y)
        dT_d = np.dot(d.T, d)
        eigen_val = np.linalg.eigvalsh(dT_d)

        # Initialize variance and precision using Evidence approximation
        model = pm.Model()
        with model:
            p = pm.Gamma('p', alpha=self.alpha_1, beta=self.alpha_2)
            v = pm.Gamma('v', alpha=self.alpha_3, beta=self.alpha_4)
            b = pm.Normal('b', mu=0, tau=p, shape=(len(d[0]), n_class))
            y_obs = pm.Normal('y_obs', mu=pm.math.dot(d, b), tau=v, observed=y)
        
        map_estimate =  pm.find_MAP(model=model)
        self.prec, self.var, self.beta = map_estimate['p'].item(0), map_estimate['v'].item(0), map_estimate['b']
        # print('Initialized parameters:')
        # print('precision: ', self.prec)
        # print('variance: ', self.var)
        # print('beta: ', self.beta)

        # Iterate to meet convergence criteria
        mean_prev = None
        for iter_ in range(self.n_iter):
            # Posterior update
            # update posterior covariance
            self.covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
            # update posterior mean
            self.mean = np.dot(self.covar, dT_y) / self.var

            # Hyperparameters update
            # update eigenvalues
            lam = eigen_val / self.var
            # update precision and variance 
            delta = np.sum(np.divide(lam, lam + self.prec))
            self.prec = (delta + 2 * self.alpha_1) / (np.sum(np.square(self.mean)) + 2 * self.alpha_2)
            self.var = (np.sum(np.square(y - np.dot(d, self.beta))) + self.alpha_4) / (n_sample + delta + 2 * self.alpha_3)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(mean_prev - self.mean)) < self.tol:
                print("Convergence after ", str(iter_), " iterations")
                break
            mean_prev = np.copy(self.mean)

        # Final Posterior update
        # update posterior covariance
        self.covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
        # update posterior mean
        self.beta = np.dot(self.covar, dT_y) / self.var
        # print('beta: ', np.shape(self.beta))
        # print('Posterior mean: ', self.mean)
        # print('Posterior covariance: ', self.covar)
        # print('Precision: ', self.prec)
        # print('Variance: ', self.var)

    def predict(self, data, raw_output=False):
        n_sample = len(data)
        d = self.standardize(data, 0) # Normalize
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, d], axis=1)

        # d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        result = self.softmax(np.dot(d, self.beta))
        if not raw_output:
            result = np.argmax(result, axis=1)
        return result

    def eval(self, data, label):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1
        
        result = self.predict(data, False)
        acc = np.sum(np.equal(result, label))/len(label)
        return acc

    def one_hot_encoding(self, label, n_class):
        y = np.zeros([len(label), n_class])
        for i in range(len(label)):
            y[i, label[i]] = 1
        return y

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]

    def softmax(self, x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)

class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))
    
    def sine(self, x):
        return np.sin(x)
    
    def sign(self, x):
        return np.sign(x)
    
    def relu(self, x):
        return np.maximum(0, x)

if __name__=="__main__":
    dataset = loadmat('coil20.mat')
    label = np.array([dataset['Y'][i][0] - 1 for i in range(len(dataset['Y']))])
    data = dataset['X']
    n_class = 20

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    val_acc = []
    max_index = -1
    
    for i, kf_values in enumerate(kf.split(X_train, y_train)):
    #     print(f'train: {train_index}, val: {val_index}')
        print('Validation: {}'.format(i + 1))
        train_index, val_index = kf_values
        X_val_train, X_val_test = X_train[train_index], X_train[val_index]
        y_val_train, y_val_test = y_train[train_index], y_train[val_index]
        bdrvfl = BayesianDeepRVFL(n_node, w_range, b_range, n_layer)
        bdrvfl.train(X_val_train, y_val_train, n_class)
        prediction = bdrvfl.predict(X_val_test, True)
        acc = bdrvfl.eval(X_val_test, y_val_test)
        print(f'Validation accuracy: {acc}')
        val_acc.append(acc)
        if acc >= max(val_acc):
            max_index = train_index

    X_train, y_train = X_train[max_index], y_train[max_index]
    print(np.shape(X_train))
    bdrvfl = BayesianDeepRVFL(n_node, w_range, b_range, n_layer)
    bdrvfl.train(X_train, y_train, n_class)
    prediction = bdrvfl.predict(X_test, True)
    acc = bdrvfl.eval(X_test, y_test)
    print(f'\nTest accuracy: {acc}')