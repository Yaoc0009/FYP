import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.io import loadmat
import pymc3 as pm

n_node = 10 # num of nodes in hidden layer
w_range = [-1, 1] # range of random weights
b_range = [0, 1] # range of random biases

class BRVFL:
    """ BRVFL Classifier """

    def __init__(self, n_node, w_range, b_range, alpha_1=10**(-5), alpha_2=10**(-5), alpha_3=10**(-5), alpha_4=10**(-5), n_iter=1000, tol=1.0e-3, activation='relu', same_feature=False):
        self.n_node = n_node
        self.w_range = w_range
        self.b_range = b_range
        
        self.alpha_1 = alpha_1 # Gamma distribution parameter
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.n_iter = n_iter
        self.tol = tol

        self.weight = None
        self.bias = None
        self.beta = None
        self.prec = None
        self.var = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature

    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1

        data = self.standardize(data)
        n_sample, n_feature = np.shape(data)
        self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
        self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
        h = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
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

        # Iterate to meet convergence criteria
        mean_prev = None
        for iter_ in range(self.n_iter):
            # Posterior update
            # update posterior covariance
            covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
            # update posterior mean
            mean = np.dot(covar, dT_y) / self.var

            # Hyperparameters update
            # update eigenvalues
            lam = eigen_val / self.var
            # update precision and variance 
            delta = np.sum(np.divide(lam, lam + self.prec))
            self.prec = (delta + 2 * self.alpha_1) / (np.sum(np.square(mean)) + 2 * self.alpha_2)
            self.var = (np.sum(np.square(y - np.dot(d, self.beta))) + self.alpha_4) / (n_sample + delta + 2 * self.alpha_3)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(mean_prev - mean)) < self.tol:
                print("Convergence after ", str(iter_), " iterations")
                break
            mean_prev = np.copy(mean)

        # Final Posterior update
        # update posterior covariance
        covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
        # update posterior mean
        self.beta = np.dot(covar, dT_y) / self.var

    def predict(self, data, raw_output=False):
        data = self.standardize(data) # Normalize
        h = self.activation_function(np.dot(data, self.weight) + self.bias)
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
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

    def standardize(self, x):
        if self.same_feature is True:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if self.data_std is None:
                self.data_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean is None:
                self.data_mean = np.mean(x, axis=0)
            return (x - self.data_mean) / self.data_std

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
    n_class = len(np.unique(label))

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
        brvfl = BRVFL(n_node, w_range, b_range)
        brvfl.train(X_val_train, y_val_train, n_class)
        prediction = brvfl.predict(X_val_test, True)
        acc = brvfl.eval(X_val_test, y_val_test)
        print(f'Validation accuracy: {acc}')
        val_acc.append(acc)
        if acc >= max(val_acc):
            max_index = train_index

    X_train, y_train = X_train[max_index], y_train[max_index]
    brvfl = BRVFL(n_node, w_range, b_range)
    brvfl.train(X_train, y_train, n_class)
    prediction = brvfl.predict(X_test, True)
    acc = brvfl.eval(X_test, y_test)
    print(f'\nTest accuracy: {acc}')