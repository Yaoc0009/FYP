from sklearn.model_selection import KFold
import numpy as np
from models import *
from time import time

# set random seed
np.random.seed(42)

lams = [2**i for i in range(-6, 13, 2)] # regularization parameter, lambda

def cross_val_acc(data, label, n_class, model_class, lam=None, n_layer=1, activation='sigmoid'):
    n_node = 100 # num of nodes in hidden layer
    w_range = [-1, 1] # range of random weights
    b_range = [0, 1] # range of random biases
    val_acc = []
    train_time = []
    test_time = []

    kf = KFold(n_splits=10, shuffle=True)
    # print('Lambda: ', lam)
    for _, kf_values in enumerate(kf.split(data, label)):
        # print('Validation: {}'.format(i + 1))
        train_index, val_index = kf_values
        X_val_train, X_val_test = data[train_index], data[val_index]
        y_val_train, y_val_test = label[train_index], label[val_index]
        if model_class in [RVFL, DeepRVFL, EnsembleDeepRVFL]:
            model = model_class(n_node, lam, w_range, b_range, n_layer, activation=activation)
        elif model_class in [BRVFL, BDeepRVFL, BEnsembleDeepRVFL]:
            model = model_class(n_node, w_range, b_range, n_layer, tol=10**(-7), activation=activation)
        t = time()
        model.train(X_val_train, y_val_train, n_class)
        train_t = time()
        acc = model.eval(X_val_test, y_val_test)
        test_t = time()
        train_time.append(train_t - t)
        test_time.append(test_t - train_t)

        # print(f'Validation accuracy: {acc}')
        val_acc.append(acc)

    acc = [np.mean(val_acc), np.std(val_acc)]
    t = [np.mean(train_time), np.mean(test_time)]
    # print(f'Model accuracy: {mean_acc}\n')
    return model, acc, t

def run_all(data, label, n_class):
    run_RVFL(data, label, n_class)
    run_dRVFL(data, label, n_class)
    run_edRVFL(data, label, n_class)
    run_BRVFL(data, label, n_class)
    run_BdRVFL(data, label, n_class)
    run_BedRVFL(data, label, n_class)

def run_RVFL(data, label, n_class):
    print('running RVFL...')
    acc = []
    t = []
    for lam in lams:
        _, model_accuracy, duration = cross_val_acc(data, label, n_class, RVFL, lam)
        acc.append(model_accuracy)
        t.append(duration)

    max_index = np.argmax(acc, axis=0)[0]
    opt_lam = lams[max_index]
    print('Accuracy: ', acc[max_index][0], u"\u00B1", acc[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index][0])
    print('Test time: ', t[max_index][1])

def run_dRVFL(data, label, n_class):
    print('running Deep RVFL...')
    acc = []
    t = []
    for lam in lams:
        _, model_accuracy, duration = cross_val_acc(data, label, n_class, DeepRVFL, lam, n_layer=5)
        acc.append(model_accuracy)
        t.append(duration)

    max_index = np.argmax(acc, axis=0)[0]
    opt_lam = lams[max_index]
    print('Accuracy: ', acc[max_index][0], u"\u00B1", acc[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index][0])
    print('Test time: ', t[max_index][1])

def run_edRVFL(data, label, n_class):
    print('running Ensemble Deep RVFL...')
    acc = []
    t = []
    for lam in lams:
        _, model_accuracy, duration = cross_val_acc(data, label, n_class, EnsembleDeepRVFL, lam, n_layer=5)
        acc.append(model_accuracy)
        t.append(duration)

    max_index = np.argmax(acc, axis=0)[0]
    opt_lam = lams[max_index]
    print('Accuracy: ', acc[max_index][0], u"\u00B1", acc[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index][0])
    print('Test time: ', t[max_index][1])

def run_BRVFL(data, label, n_class):
    print('running Bayesian RVFL...')
    model, acc, t = cross_val_acc(data, label, n_class, BRVFL)

    print('Accuracy: ', acc[0], u"\u00B1", acc[1])
    print('Train time: ', t[0])
    print('Test time: ', t[1])
    print('Hyperparameters: ')
    print('Precision: ', model.prec)
    print('Variance: ', model.var)

def run_BdRVFL(data, label, n_class):
    print('running Bayesian Deep RVFL...')
    model, acc, t = cross_val_acc(data, label, n_class, BDeepRVFL, n_layer=5)

    print('Accuracy: ', acc[0], u"\u00B1", acc[1])
    print('Train time: ', t[0])
    print('Test time: ', t[1])
    print('Hyperparameters: ')
    print('Precision: ', model.prec)
    print('Variance: ', model.var)

def run_BedRVFL(data, label, n_class):
    print('running Bayesian Ensemble Deep RVFL...')
    model, acc, t = cross_val_acc(data, label, n_class, BEnsembleDeepRVFL, n_layer=5)

    print('Accuracy: ', acc[0], u"\u00B1", acc[1])
    print('Train time: ', t[0])
    print('Test time: ', t[1])
    print('Hyperparameters: ')
    print('Precision: ', model.prec)
    print('Variance: ', model.var)

