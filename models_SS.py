import numpy as np
# import pymc3 as pm

# set random seed
np.random.seed(42)
np.seterr(over='ignore')

class Model:
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
        return np.exp(x - np.max(x)) / np.repeat((np.sum(np.exp(x-np.max(x)), axis=1))[:, np.newaxis], len(x[0]), axis=1)

    def beta_function(self, d, L, C0, lam, y, label, label_proportions):
        # number of samples per class label
        n_sample, n_feature = np.shape(d)
        C = np.zeros_like(L)
        for i, class_num in enumerate(label):
            C[i, i] = float(C0)/label_proportions[class_num]
        # More labeled examples than hidden neurons
        if n_sample > n_feature:
            I = np.identity(n_feature)
            inv_arg = I + np.linalg.multi_dot([d.T, C, d]) + lam * np.linalg.multi_dot([d.T, L, d])
            beta = np.linalg.multi_dot([np.linalg.inv(inv_arg), d.T, C, y])
        # Less labeled examples than hidden neurons (apparently the more common case)
        else:
            I = np.identity(n_sample)
            inv_arg = I + np.linalg.multi_dot([C, d, d.T]) + lam * np.linalg.multi_dot([L, d, d.T])
            beta = np.linalg.multi_dot([d.T, np.linalg.inv(inv_arg), C, y])
        beta = np.asarray(beta)
        return beta

class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))
    
    def sine(self, x):
        return np.sin(x)
    
    def sign(self, x):
        return np.sign(x)
    
    def relu(self, x):
        return np.maximum(0, x)

class ELM(Model):
    """ ELM Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, n_layer=1, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.n_layer = n_layer
        self.lam = lam
        self.w_range = w_range
        self.b_range = b_range
        self.weight = None
        self.bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        
    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1

        data = self.standardize(data) # Normalize
        n_sample = len(data)
        n_feature = len(data[0])
        self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
        self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
        d = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
        y = self.one_hot_encoding(label, n_class)
        # Minimize training complexity
        if n_sample > self.n_node:
            self.beta = (np.linalg.inv(self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)
            
    def predict(self, data, raw_output=False):
        data = self.standardize(data) # Normalize
        d = self.activation_function(np.dot(data, self.weight) + self.bias)
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
        return acc, result

class RVFL(Model):
    """ RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, n_layer=1, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.n_layer = n_layer
        self.lam = lam
        self.w_range = w_range
        self.b_range = b_range
        self.weight = None
        self.bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        
    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1

        data = self.standardize(data) # Normalize
        n_sample = len(data)
        n_feature = len(data[0])
        self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
        self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
        h = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
        d = np.concatenate([h, data], axis=1)
        y = self.one_hot_encoding(label, n_class)
        # Minimize training complexity
        if n_sample > len(d[0]):
            self.beta = (np.linalg.inv(self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)
            
    def predict(self, data, raw_output=False):
        data = self.standardize(data) # Normalize
        h = self.activation_function(np.dot(data, self.weight) + self.bias)
        d = np.concatenate([h, data], axis=1)
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
        return acc, result

class DeepRVFL(Model):
    """ Deep RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, n_layer, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.lam = lam
        self.w_range = w_range
        self.b_range = b_range
        self.weight = []
        self.bias = []
        self.beta = None
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
        d = self.standardize(data, 0) # Normalize
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
            self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, d], axis=1)

        y = self.one_hot_encoding(label, n_class)
        # Minimize training complexity
        if n_sample > len(d[0]):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)
            
    def predict(self, data, raw_output=False):
        n_sample = len(data)
        d = self.standardize(data, 0) # Normalize
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, d], axis=1)

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
        return acc, result

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
        
class EnsembleDeepRVFL(Model):
    """ Ensemble Deep RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, n_layer, activation='sigmoid', same_feature=True):
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

            # Minimize training complexity
            if n_sample > len(d[0]):
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
        return acc, result

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

class LapELM(Model):
    """ Laplacian ELM Classifier """
    def __init__(self, n_node, lam, w_range, b_range, NN, L, n_layer=1, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.n_layer = n_layer
        self.NN = NN
        self.L = L
        self.C0 = 1 / lam[0]
        self.lam = lam[1]
        self.w_range = w_range
        self.b_range = b_range
        self.weight = None
        self.bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature

    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1
        
        data = self.standardize(data) # Normalize
        n_sample = len(data)
        n_feature = len(data[0])
        self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
        self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
        d = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
        y = self.one_hot_encoding(label, n_class)
        label_proportions = np.sum(y, axis=0)
        assert sum(label_proportions) == len(label)
        self.beta = self.beta_function(d, self.L, self.C0, self.lam, y, label, label_proportions)
            
    def predict(self, data, raw_output=False):
        data = self.standardize(data) # Normalize
        d = self.activation_function(np.dot(data, self.weight) + self.bias)
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
        return acc, result

class LapRVFL(Model):
    """ Laplacian RVFL Classifier """
    def __init__(self, n_node, lam, w_range, b_range, NN, L, n_layer=1, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.n_layer = n_layer
        self.NN = NN
        self.L = L
        self.C0 = 1 / lam[0]
        self.lam = lam[1]
        self.w_range = w_range
        self.b_range = b_range
        self.weight = None
        self.bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        
    def train(self, data, label, n_class):
        assert len(data.shape) > 1
        assert len(data) == len(label)
        assert len(label.shape) == 1
        
        data = self.standardize(data) # Normalize
        n_sample = len(data)
        n_feature = len(data[0])
        self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
        self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
        h = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
        d = np.concatenate([h, data], axis=1)
        y = self.one_hot_encoding(label, n_class)
        label_proportions = np.sum(y, axis=0)
        assert sum(label_proportions) == len(label)
        self.beta = self.beta_function(d, self.L, self.C0, self.lam, y, label, label_proportions)
            
    def predict(self, data, raw_output=False):
        data = self.standardize(data) # Normalize
        h = self.activation_function(np.dot(data, self.weight) + self.bias)
        d = np.concatenate([h, data], axis=1)
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
        return acc, result

class LapDeepRVFL(Model):
    """ Laplacian Deep RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, NN, L, n_layer, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.NN = NN
        self.L = L
        self.C0 = 1 / lam[0]
        self.lam = lam[1]
        self.w_range = w_range
        self.b_range = b_range
        self.weight = []
        self.bias = []
        self.beta = None
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
        d = self.standardize(data, 0) # Normalize
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
            self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, d], axis=1)

        y = self.one_hot_encoding(label, n_class)
        label_proportions = np.sum(y, axis=0)
        assert sum(label_proportions) == len(label)
        self.beta = self.beta_function(d, self.L, self.C0, self.lam, y, label, label_proportions)
            
    def predict(self, data, raw_output=False):
        n_sample = len(data)
        d = self.standardize(data, 0) # Normalize
        h = data.copy()
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, d], axis=1)

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
        return acc, result

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
        
class LapEnsembleDeepRVFL(Model):
    """ Laplacian Ensemble Deep RVFL Classifier """
    
    def __init__(self, n_node, lam, w_range, b_range, NN, L, n_layer, activation='sigmoid', same_feature=True):
        self.n_node = n_node
        self.NN = NN
        self.L = L
        self.C0 = 1 / lam[0]
        self.lam = lam[1]
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
        label_proportions = np.sum(y, axis=0)
        assert sum(label_proportions) == len(label)
        for i in range(self.n_layer):
            h = self.standardize(h, i)
            self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
            self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
            h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
            d = np.concatenate([h, data], axis=1)
            h = d
            self.beta.append(self.beta_function(d, self.L, self.C0, self.lam, y, label, label_proportions))
            
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
        return acc, result

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

# class BRVFL2(Model):
#     """ BRVFL Classifier """

#     def __init__(self, n_node, prec, w_range, b_range, n_layer=1, alpha_1=10**(-5), alpha_2=10**(-5), alpha_3=10**(-5), alpha_4=10**(-3), n_iter=1000, tol=1.0e-3, activation='sigmoid', same_feature=True):
#         self.n_node = n_node
#         self.w_range = w_range
#         self.b_range = b_range
        
#         self.alpha_1 = alpha_1 # Gamma distribution parameter
#         self.alpha_2 = alpha_2
#         self.alpha_3 = alpha_3
#         self.alpha_4 = alpha_4
#         self.n_iter = n_iter
#         self.tol = tol

#         self.model = None
#         self.weight = None
#         self.bias = None
#         self.beta = None
#         self.prec = prec
#         self.var = 1
#         a = Activation()
#         self.activation_function = getattr(a, activation)
#         self.data_std = None
#         self.data_mean = None
#         self.same_feature = same_feature

#     def train(self, data, label, n_class):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1

#         data = self.standardize(data)
#         n_sample, n_feature = np.shape(data)
#         self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
#         self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
#         h = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
#         d = np.concatenate([h, data], axis=1)
#         y = self.one_hot_encoding(label, n_class)
#         dT_y = np.dot(d.T, y)
#         dT_d = np.dot(d.T, d)
#         eigen_val = np.linalg.eigvalsh(dT_d)

#         # Initialize variance and precision using Evidence approximation
#         self.model = pm.Model()
#         with self.model:
#             # build model
#             precision = pm.Gamma('precision', alpha=self.alpha_1, beta=self.alpha_2)
#             variance = pm.Gamma('variance', alpha=self.alpha_3, beta=self.alpha_4)
#             beta = pm.Normal('beta', mu=0, tau=precision, shape=(len(d[0]), n_class))
#             y_obs = pm.Normal('y_obs', mu=pm.math.dot(d, beta), tau=variance, observed=y)
        
#         map_estimate =  pm.find_MAP(model=self.model, progressbar=False)
#         # self.prec, self.var, self.beta = map_estimate['precision'].item(0), map_estimate['variance'].item(0), map_estimate['beta']
#         self.beta = map_estimate['beta']
#         # Iterate to meet convergence criteria
#         mean_prev = None
#         for iter_ in range(self.n_iter):
#             # Posterior update
#             # update posterior covariance
#             covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
#             # update posterior mean
#             mean = np.dot(covar, dT_y) / self.var

#             # Hyperparameters update
#             # update eigenvalues
#             lam = eigen_val / self.var
#             # update precision and variance 
#             delta = np.sum(np.divide(lam, lam + self.prec))
#             self.prec = (delta + 2 * self.alpha_1) / (np.sum(np.square(mean)) + 2 * self.alpha_2)
#             self.var = (np.sum(np.square(y - np.dot(d, self.beta))) + self.alpha_4) / (n_sample + delta + 2 * self.alpha_3)

#             # Check for convergence
#             if iter_ != 0 and np.sum(np.abs(mean_prev - mean)) < self.tol:
#                 print("Convergence after ", str(iter_), " iterations")
#                 break
#             mean_prev = np.copy(mean)

#         # Final Posterior update
#         # update posterior covariance
#         covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
#         # update posterior mean
#         self.beta = np.dot(covar, dT_y) / self.var

#     def predict(self, data, raw_output=False):
#         data = self.standardize(data) # Normalize
#         h = self.activation_function(np.dot(data, self.weight) + self.bias)
#         d = np.concatenate([h, data], axis=1)
#         result = self.softmax(np.dot(d, self.beta))
#         if not raw_output:
#             result = np.argmax(result, axis=1)
#         return result

#     def eval(self, data, label):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1
        
#         result = self.predict(data, False)
#         acc = np.sum(np.equal(result, label))/len(label)
#         return acc, result

# class BRVFL(Model):
#     """ BRVFL Classifier """

#     def __init__(self, n_node, w_range, b_range, n_layer=1, alpha_1=10**(-5), alpha_2=10**(-5), alpha_3=10**(-5), alpha_4=10**(-5), n_iter=1000, tol=1.0e-3, activation='sigmoid', same_feature=True):
#         self.n_node = n_node
#         self.w_range = w_range
#         self.b_range = b_range
#         self.n_layer = n_layer
        
#         self.alpha_1 = alpha_1 # Gamma distribution parameter
#         self.alpha_2 = alpha_2
#         self.alpha_3 = alpha_3
#         self.alpha_4 = alpha_4
#         self.n_iter = n_iter
#         self.tol = tol

#         self.weight = None
#         self.bias = None
#         self.beta = None
#         self.prec = None
#         self.var = None
#         a = Activation()
#         self.activation_function = getattr(a, activation)
#         self.data_std = None
#         self.data_mean = None
#         self.same_feature = same_feature

#     def train(self, data, label, n_class):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1

#         data = self.standardize(data)
#         n_sample, n_feature = np.shape(data)
#         self.weight = (self.w_range[1] - self.w_range[0]) * np.random.random([n_feature, self.n_node]) + self.w_range[0]
#         self.bias = (self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0]
        
#         h = self.activation_function(np.dot(data, self.weight) + np.dot(np.ones([n_sample, 1]), self.bias))
#         d = np.concatenate([h, data], axis=1)
#         y = self.one_hot_encoding(label, n_class)
#         dT_y = np.dot(d.T, y)
#         dT_d = np.dot(d.T, d)
#         eigen_val = np.linalg.eigvalsh(dT_d)

#         # Initialize variance and precision using Evidence approximation
#         model = pm.Model()
#         with model:
#             p = pm.Gamma('p', alpha=self.alpha_1, beta=self.alpha_2)
#             v = pm.Gamma('v', alpha=self.alpha_3, beta=self.alpha_4)
#             b = pm.Normal('b', mu=0, tau=p, shape=(len(d[0]), n_class))
#             y_obs = pm.Normal('y_obs', mu=pm.math.dot(d, b), tau=v, observed=y)
        
#         map_estimate =  pm.find_MAP(model=model, progressbar=False)
#         self.prec, self.var, self.beta = map_estimate['p'].item(0), map_estimate['v'].item(0), map_estimate['b']

#         # Iterate to meet convergence criteria
#         mean_prev = None
#         for iter_ in range(self.n_iter):
#             # Posterior update
#             # update posterior covariance
#             covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
#             # update posterior mean
#             mean = np.dot(covar, dT_y) / self.var

#             # Hyperparameters update
#             # update eigenvalues
#             lam = eigen_val / self.var
#             # update precision and variance 
#             delta = np.sum(np.divide(lam, lam + self.prec))
#             self.prec = (delta + 2 * self.alpha_1) / (np.sum(np.square(mean)) + 2 * self.alpha_2)
#             self.var = (np.sum(np.square(y - np.dot(d, self.beta))) + self.alpha_4) / (n_sample + delta + 2 * self.alpha_3)

#             # Check for convergence
#             if iter_ != 0 and np.sum(np.abs(mean_prev - mean)) < self.tol:
#                 print("Convergence after ", str(iter_), " iterations")
#                 break
#             mean_prev = np.copy(mean)

#         # Final Posterior update
#         # update posterior covariance
#         covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
#         # update posterior mean
#         self.beta = np.dot(covar, dT_y) / self.var

#     def predict(self, data, raw_output=False):
#         data = self.standardize(data) # Normalize
#         h = self.activation_function(np.dot(data, self.weight) + self.bias)
#         d = np.concatenate([h, data], axis=1)
#         result = self.softmax(np.dot(d, self.beta))
#         if not raw_output:
#             result = np.argmax(result, axis=1)
#         return result

#     def eval(self, data, label):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1
        
#         result = self.predict(data, False)
#         acc = np.sum(np.equal(result, label))/len(label)
#         return acc, result

# class BDeepRVFL(Model):
#     """ Bayesian Deep RVFL Classifier """

#     def __init__(self, n_node, w_range, b_range, n_layer, alpha_1=10**(-5), alpha_2=10**(-5), alpha_3=10**(-5), alpha_4=10**(-5), n_iter=1000, tol=1.0e-3, activation='sigmoid', same_feature=True):
#         self.n_node = n_node
#         self.w_range = w_range
#         self.b_range = b_range
#         self.n_layer = n_layer
        
#         self.alpha_1 = alpha_1 # Gamma distribution parameters
#         self.alpha_2 = alpha_2
#         self.alpha_3 = alpha_3
#         self.alpha_4 = alpha_4
#         self.n_iter = n_iter
#         self.tol = tol

#         self.weight = []
#         self.bias = []
#         self.beta = None
#         self.prec = None
#         self.var = None
#         a = Activation()
#         self.activation_function = getattr(a, activation)
#         self.data_std = [None] * self.n_layer
#         self.data_mean = [None] * self.n_layer
#         self.same_feature = same_feature

#     def train(self, data, label, n_class):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1

#         n_sample, n_feature = np.shape(data)
#         d = self.standardize(data, 0) # Normalize
#         h = data.copy()
#         for i in range(self.n_layer):
#             h = self.standardize(h, i)
#             self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
#             self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
#             h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
#             d = np.concatenate([h, d], axis=1)

#         y = self.one_hot_encoding(label, n_class)
#         dT_y = np.dot(d.T, y)
#         dT_d = np.dot(d.T, d)
#         eigen_val = np.linalg.eigvalsh(dT_d)

#         # Initialize variance and precision using Evidence approximation
#         model = pm.Model()
#         with model:
#             p = pm.Gamma('p', alpha=self.alpha_1, beta=self.alpha_2)
#             v = pm.Gamma('v', alpha=self.alpha_3, beta=self.alpha_4)
#             b = pm.Normal('b', mu=0, tau=p, shape=(len(d[0]), n_class))
#             y_obs = pm.Normal('y_obs', mu=pm.math.dot(d, b), tau=v, observed=y)
        
#         map_estimate =  pm.find_MAP(model=model, progressbar=False)
#         self.prec, self.var, self.beta = map_estimate['p'].item(0), map_estimate['v'].item(0), map_estimate['b']

#         # Iterate to meet convergence criteria
#         mean_prev = None
#         for iter_ in range(self.n_iter):
#             # Posterior update
#             # update posterior covariance
#             covar = np.linalg.inv(self.prec * np.identity(dT_d.shape[1]) + dT_d / self.var)
#             # update posterior mean
#             mean = np.dot(covar, dT_y) / self.var

#             # Hyperparameters update
#             # update eigenvalues
#             lam = eigen_val / self.var
#             # update precision and variance 
#             delta = np.sum(np.divide(lam, lam + self.prec))
#             self.prec = (delta + 2 * self.alpha_1) / (np.sum(np.square(mean)) + 2 * self.alpha_2)
#             self.var = (np.sum(np.square(y - np.dot(d, self.beta))) + self.alpha_4) / (n_sample + delta + 2 * self.alpha_3)

#             # Check for convergence
#             if iter_ != 0 and np.sum(np.abs(mean_prev - mean)) < self.tol:
#                 print("Convergence after ", str(iter_), " iterations")
#                 break
#             mean_prev = np.copy(mean)

#         # Final Posterior update
#         # update posterior covariance
#         covar = np.linalg.inv(self.prec * np.identity(d.shape[1]) + dT_d / self.var)
#         # update posterior mean
#         self.beta = np.dot(covar, dT_y) / self.var

#     def predict(self, data, raw_output=False):
#         n_sample = len(data)
#         d = self.standardize(data, 0) # Normalize
#         h = data.copy()
#         for i in range(self.n_layer):
#             h = self.standardize(h, i)
#             h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
#             d = np.concatenate([h, d], axis=1)

#         result = self.softmax(np.dot(d, self.beta))
#         if not raw_output:
#             result = np.argmax(result, axis=1)
#         return result

#     def eval(self, data, label):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1
        
#         result = self.predict(data, False)
#         acc = np.sum(np.equal(result, label))/len(label)
#         return acc, result

#     def standardize(self, x, index):
#         if self.same_feature is True:
#             if self.data_std[index] is None:
#                 self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
#             if self.data_mean[index] is None:
#                 self.data_mean[index] = np.mean(x)
#             return (x - self.data_mean[index]) / self.data_std[index]
#         else:
#             if self.data_std[index] is None:
#                 self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
#             if self.data_mean[index] is None:
#                 self.data_mean[index] = np.mean(x, axis=0)
#             return (x - self.data_mean[index]) / self.data_std[index]

# class BEnsembleDeepRVFL(Model):
#     """ Bayesian Deep RVFL Classifier """

#     def __init__(self, n_node, w_range, b_range, n_layer, alpha_1=10**(-5), alpha_2=10**(-5), alpha_3=10**(-5), alpha_4=10**(-5), n_iter=1000, tol=1.0e-3, activation='sigmoid', same_feature=True):
#         self.n_node = n_node
#         self.w_range = w_range
#         self.b_range = b_range
#         self.n_layer = n_layer
        
#         self.alpha_1 = alpha_1 # Gamma distribution parameters
#         self.alpha_2 = alpha_2
#         self.alpha_3 = alpha_3
#         self.alpha_4 = alpha_4
#         self.n_iter = n_iter
#         self.tol = tol

#         self.weight = []
#         self.bias = []
#         self.beta = []
#         self.prec = []
#         self.var = []
#         a = Activation()
#         self.activation_function = getattr(a, activation)
#         self.data_std = [None] * self.n_layer
#         self.data_mean = [None] * self.n_layer
#         self.same_feature = same_feature

#     def train(self, data, label, n_class):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1

#         n_sample, n_feature = np.shape(data)
#         data = self.standardize(data, 0) # Normalize
#         h = data.copy()
#         y = self.one_hot_encoding(label, n_class)
#         for i in range(self.n_layer):
#             h = self.standardize(h, i)
#             self.weight.append((self.w_range[1] - self.w_range[0]) * np.random.random([len(h[0]), self.n_node]) + self.w_range[0])
#             self.bias.append((self.b_range[1] - self.b_range[0]) * np.random.random([1, self.n_node]) + self.b_range[0])
#             h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
#             d = np.concatenate([h, data], axis=1)
#             h = d

#             dT_y = np.dot(d.T, y)
#             dT_d = np.dot(d.T, d)
#             eigen_val = np.linalg.eigvalsh(dT_d)

#             # initial map estimate 
#             if i == 0:
#                 # Initialize variance and precision using Evidence approximation
#                 model = pm.Model()
#                 with model:
#                     p = pm.Gamma('p', alpha=self.alpha_1, beta=self.alpha_2)
#                     v = pm.Gamma('v', alpha=self.alpha_3, beta=self.alpha_4)
#                     b = pm.Normal('b', mu=0, tau=p, shape=(len(d[0]), n_class))
#                     y_obs = pm.Normal('y_obs', mu=pm.math.dot(d, b), tau=v, observed=y)
                
#                 map_estimate =  pm.find_MAP(model=model, progressbar=False)
#                 # Set map estimate of prec, var, beta as initial value for each model in the ensemble
#                 self.prec = [map_estimate['p'].item(0) for _ in range(self.n_layer)]
#                 self.var = [map_estimate['v'].item(0) for _ in range(self.n_layer)]
#                 self.beta = [map_estimate['b'] for _ in range(self.n_layer)]
            
#             # Iterate to meet convergence criteria
#             mean_prev = None
#             for iter_ in range(self.n_iter):
#                 # Posterior update
#                 # update posterior covariance
#                 covar = np.linalg.inv(self.prec[i] * np.identity(d.shape[1]) + dT_d / self.var[i])
#                 # update posterior mean
#                 mean = np.dot(covar, dT_y) / self.var[i]

#                 # Hyperparameters update
#                 # update eigenvalues
#                 lam = eigen_val / self.var[i]
#                 # update precision and variance 
#                 delta = np.sum(np.divide(lam, lam + self.prec[i]))
#                 self.prec[i] = (delta + 2 * self.alpha_1) / (np.sum(np.square(mean)) + 2 * self.alpha_2)
#                 self.var[i] = (np.sum(np.square(y - np.dot(d, self.beta[i]))) + self.alpha_4) / (n_sample + delta + 2 * self.alpha_3)

#                 # Check for convergence
#                 if iter_ != 0 and np.sum(np.abs(mean_prev - mean)) < self.tol:
#                     print("Convergence after ", str(iter_), " iterations")
#                     break
#                 mean_prev = np.copy(mean)

#             # Final Posterior update
#             # update posterior covariance
#             covar = np.linalg.inv(self.prec[i] * np.identity(dT_d.shape[1]) + dT_d / self.var[i])
#             # update posterior mean
#             self.beta[i] = np.dot(covar, dT_y) / self.var[i]

#     def predict(self, data, raw_output=False):
#         n_sample = len(data)
#         data = self.standardize(data, 0) # Normalize
#         h = data.copy()
#         results = []
#         for i in range(self.n_layer):
#             h = self.standardize(h, i)
#             h = self.activation_function(np.dot(h, self.weight[i]) + np.dot(np.ones([n_sample, 1]), self.bias[i]))
#             d = np.concatenate([h, data], axis=1)
#             h = d

#             if not raw_output:
#                 results.append(np.argmax(np.dot(d, self.beta[i]), axis=1))
#             else:
#                 results.append(self.softmax(np.dot(d, self.beta[i])))

#         if not raw_output:
#             results = list(map(np.bincount, list(np.array(results).transpose())))
#             results = np.array(list(map(np.argmax, results)))
#         return results

#     def eval(self, data, label):
#         assert len(data.shape) > 1
#         assert len(data) == len(label)
#         assert len(label.shape) == 1
        
#         result = self.predict(data, False)
#         acc = np.sum(np.equal(result, label))/len(label)
#         return acc, result
    
#     def standardize(self, x, index):
#         if self.same_feature is True:
#             if self.data_std[index] is None:
#                 self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
#             if self.data_mean[index] is None:
#                 self.data_mean[index] = np.mean(x)
#             return (x - self.data_mean[index]) / self.data_std[index]
#         else:
#             if self.data_std[index] is None:
#                 self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
#             if self.data_mean[index] is None:
#                 self.data_mean[index] = np.mean(x, axis=0)
#             return (x - self.data_mean[index]) / self.data_std[index]
