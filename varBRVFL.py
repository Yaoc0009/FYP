import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.io import loadmat
import pymc3 as pm
import theano.tensor as T
import matplotlib.pyplot as plt

n_node = 10 # num of nodes in hidden layer
w_range = [-1, 1] # range of random weights
b_range = [0, 1] # range of random biases

class VariationalBRVFL:
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

        self.model = None
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
    
    def kernel(self, data, beta, weights, bias):
        h = self.activation_function(T.dot(data, weights) + bias)
        d = T.concatenate([h, data], axis=1)
        d = T.concatenate([d, T.ones_like(d[:, 0:1])], axis=1)
        out = T.dot(d, beta)
        return out

    def train(self, data, label, test_data, test_label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1

        data = self.standardize(data)
        n_sample, n_feature = np.shape(data)
        y = self.one_hot_encoding(label, n_class)

        # Initialize variance and precision using Evidence approximation
        self.model = pm.Model()
        with self.model:
            bias = pm.Normal('bias', mu=0, tau=1, shape=(1, self.n_node))
            weights =pm.Normal('weights', mu=0, tau=1, shape=(n_feature, self.n_node))
            precision = pm.Gamma('precision', alpha=self.alpha_1, beta=self.alpha_2)
            variance = pm.Gamma('variance', alpha=self.alpha_3, beta=self.alpha_4)
            beta = pm.Normal('beta', mu=0, tau=precision, shape=(n_feature+self.n_node+1, n_class))
            y_obs = pm.Normal('y_obs', mu=self.kernel(data, beta, weights, bias), tau=variance, observed=y)
            start = pm.find_MAP()
            inference = pm.ADVI()
            approx = pm.fit(n=self.n_iter, start=start, method=inference)
            trace = approx.sample(draws=5000)
            pm.traceplot(trace)
            pm.summary(trace)
        
        plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
        plt.plot(approx.hist, label="old ADVI", alpha=0.3)
        plt.legend()
        plt.ylabel("ELBO")
        plt.xlabel("iteration")
            # trace = pm.sample_approx(approx=approx, draws=5000)
            # pm.traceplot(trace)
            # pm.summary(trace)
            # post_pred = pm.sample_ppc(trace, samples=5000, model=self.model)
            # y_train_pred = np.mean(post_pred['y_obs'], axis=0)
            # result_train = np.argmax(y_train_pred, axis=1)
            # train_acc = np.sum(np.equal(result_train, label))/len(label)
            # data.set_value(test_data)
            # post_pred = pm.sample_ppc(trace, samples=5000, model=self.model)
            # y_test_pred = np.mean(post_pred['y_obs'], axis=0)
            # result_test = np.argmax(y_test_pred, axis=1)
            # test_acc = np.sum(np.equal(result_test, test_label))/len(test_label)
            # return train_acc, test_acc

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
        return T.maximum(0, x)

if __name__=="__main__":
    dataset = loadmat('coil20.mat')
    label = np.array([dataset['Y'][i][0] - 1 for i in range(len(dataset['Y']))])
    data = dataset['X']
    n_class = len(np.unique(label))

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    # kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # val_accs = []
    # max_index = -1
    
    # for i, kf_values in enumerate(kf.split(X_train, y_train)):
    # #     print(f'train: {train_index}, val: {val_index}')
    #     print('Validation: {}'.format(i + 1))
    #     train_index, val_index = kf_values
    #     X_val_train, X_val_test = X_train[train_index], X_train[val_index]
    #     y_val_train, y_val_test = y_train[train_index], y_train[val_index]
    #     model = VariationalBRVFL(n_node, w_range, b_range)
    #     train_acc, val_acc = model.train(X_val_train, y_val_train, X_val_test, y_val_test, n_class)
    #     print(f'Validation accuracy: {val_acc}')
    #     val_accs.append(val_acc)
    #     if val_acc >= max(val_accs):
    #         max_index = train_index

    # X_train, y_train = X_train[max_index], y_train[max_index]
    model = VariationalBRVFL(n_node, w_range, b_range)
    train_acc, test_acc = model.train(X_train, y_train, X_test, y_test, n_class)
    print(f'\nTest accuracy: {test_acc}')