import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.io import loadmat

n_node = 10 # num of nodes in hidden layer
n_layer = 2 # num of hidden layers
lam = 1 # regularization parameter, lambda
w_range = [-1, 1] # range of random weights
b_range = [0, 1] # range of random biases

class EnsembleDeepRVFL:
    """ Ensemble Deep RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, n_layer, activation='relu', same_feature=False):
        self.n_node = n_node
        self.lam = lam
        self.w_range = w_range
        self.b_range = b_range
        self.weight = []
        self.bias = []
        self.beta = []
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature
        
    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1
        
        n_sample, n_feature = np.shape(data)
        data = self.standardize(data, 0) # Normalize
        h = data.copy()
        y = self.one_hot_encoding(label, n_class)
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
            self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1) # concat column of 1s

            # Minimize training complexity
            if n_sample > (self.n_node + n_feature):
                self.beta.append(np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y))
            else:
                self.beta.append(d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y))
            
    def predict(self, data, raw_output=False):
        n_sample = len(data)
        data = self.standardize(data, 0) # Normalize
        h = data.copy()
        results = []
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1) # concat column of 1s

            if not raw_output:
                results.append(np.argmax(np.dot(d, self.beta[i]), axis=1))
            else:
                results.append(self.softmax(np.dot(d, self.beta[i])))

        if not raw_output:
            results = list(map(np.bincount, list(np.array(results).transpose())))
            results = np.array(list(map(np.argmax, results)))
        return results
    
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
    n_class = len(np.unique(label))

    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    kf = KFold(n_splits=10, shuffle=True)
    val_acc = []
    max_index = -1
    
    for i, kf_values in enumerate(kf.split(X_train, y_train)):
    #     print(f'train: {train_index}, val: {val_index}')
        print('Validation: {}'.format(i + 1))
        train_index, val_index = kf_values
        X_val_train, X_val_test = X_train[train_index], X_train[val_index]
        y_val_train, y_val_test = y_train[train_index], y_train[val_index]
        model = EnsembleDeepRVFL(n_node, lam, w_range, b_range, n_layer)
        model.train(X_val_train, y_val_train, n_class)
        prediction = model.predict(X_val_test, False)
        acc = model.eval(X_val_test, y_val_test)
        print(f'Validation accuracy: {acc}')
        val_acc.append(acc)
        if acc >= max(val_acc):
            max_index = train_index

    X_train, y_train = X_train[max_index], y_train[max_index]
    model = EnsembleDeepRVFL(n_node, lam, w_range, b_range, n_layer)
    model.train(X_train, y_train, n_class)
    prediction = model.predict(X_test, False)
    acc = model.eval(X_test, y_test)
    print(f'\nTest accuracy: {acc}')