import numpy as np
from sklearn.model_selection import train_test_split, KFold
from scipy.io import loadmat

n_node = 10 # num of nodes in hidden layer
lam = 1 # regularization parameter, lambda
w_range = [-1, 1] # range of random weights
b_range = [0, 1] # range of random biases

class RVFL:
    """ RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, activation='relu', same_feature=False):
        self.n_node = n_node
        self.lam = lam
        self.w_range = w_range
        self.b_range = b_range
        self.weight = None
        self.bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.std = None
        self.mean = None
        self.same_feature = same_feature
        
    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1
        
        data = self.standardize(data) # Normalize
        # print('Shape of data:', np.shape(data))
        n_sample = len(data)
        n_feature = len(data[0])
        self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
        self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        # print('Shape of weights:', np.shape(self.weight))
        # print('Shape of bias:', np.shape(self.bias))
        
        h = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
        d = np.concatenate([h, data], axis=1)
        # print('Shape of D:', np.shape(d))
        # d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1) # concat column of 1s
        # print('Shape of D:', np.shape(d))
        y = self.one_hot_encoding(label, n_class)
        # print('Shape of y:', np.shape(y))
        # Minimize training complexity
        if n_sample > (self.n_node + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)
        # print('Shape of beta:', np.shape(self.beta))
            
    def predict(self, data, raw_output=False):
        data = self.standardize(data) # Normalize
        h = self.activation_function(np.dot(data, self.weight) + self.bias)
        d = np.concatenate([h, data], axis=1)
        # d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        result = self.softmax(np.dot(d, self.beta))
        if not raw_output:
            result = np.argmax(result, axis=1)
        return result
    
    def eval(self, data, label):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1
        
        data = self.standardize(data)  # Normalize
        h = self.activation_function(np.dot(data, self.weight) + self.bias)
        d = np.concatenate([h, data], axis=1)
        # d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        result = np.dot(d, self.beta)
        result = np.argmax(result, axis=1)
        acc = np.sum(np.equal(result, label))/len(label)
        return acc
        
    def one_hot_encoding(self, label, n_class):
        y = np.zeros([len(label), n_class])
        for i in range(len(label)):
            y[i, label[i]] = 1
        return y
    
    def standardize(self, x):
        if self.same_feature is True:
            if self.std is None:
                self.std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.mean is None:
                self.mean = np.mean(x)
            return (x - self.mean) / self.std
        else:
            if self.std is None:
                self.std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.mean is None:
                self.mean = np.mean(x, axis=0)
            return (x - self.mean) / self.std
        
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
        rvfl = RVFL(n_node, lam, w_range, b_range)
        rvfl.train(X_val_train, y_val_train, n_class)
        prediction = rvfl.predict(X_val_test, True)
        acc = rvfl.eval(X_val_test, y_val_test)
        print(f'Validation accuracy: {acc}')
        val_acc.append(acc)
        if acc >= max(val_acc):
            max_index = train_index

    X_train, y_train = X_train[max_index], y_train[max_index]
    rvfl = RVFL(n_node, lam, w_range, b_range)
    rvfl.train(X_train, y_train, n_class)
    prediction = rvfl.predict(X_test, True)
    acc = rvfl.eval(X_test, y_test)
    print(f'\nTest accuracy: {acc}')