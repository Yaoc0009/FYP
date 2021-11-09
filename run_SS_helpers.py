from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.utils import shuffle
import numpy as np
from models_SS import *
from datasets import *
from time import time
from Laplacian import laplacian

# set random seed
np.random.seed(42)

# hyperparameters
lams = [10**i for i in range(-6, 7)] # regularization parameter, lambda
lap_lams = [[10**i, 10**j] for i in range(-6, 7) for j in range(-6, 7)] # permutation of all possible lams

def partition_data(X_train, y_train, partition):
    L, U, V, T = partition[0], partition[1], partition[2], partition[3]
    _labeled = []
    _validation = []
    used = set()
    # fill up labeled set with one element from each class
    present = []
    for i in range(len(X_train)):
        if len(_labeled) == L:
            break
        if (y_train[i] not in present) and (i not in used):
            _labeled.append(i)
            present.append(y_train[i])
            used.add(i)
    # fill up validation set with one element from each class
    present = []
    for i in range(len(X_train)):
        if len(_validation) == V:
            break
        if (y_train[i] not in present) and (i not in used):
            _validation.append(i)
            present.append(y_train[i])
            used.add(i)
    # remaining training samples
    remaining = [i for i in range(len(X_train)) if i not in used]
    # print(len(remaining))
    lab_missing = L - len(_labeled)
    val_missing = V - len(_validation)
    # labeled
    if lab_missing <= 0:
        X_lab, y_lab = X_train[_labeled], y_train[_labeled]
        X_remain, y_remain = X_train[remaining], y_train[remaining]
    else:
        X_lab, X_remain, y_lab, y_remain = train_test_split(X_train[remaining], y_train[remaining], train_size=lab_missing)
        X_lab, y_lab = np.append(X_lab, X_train[_labeled], 0), np.append(y_lab, y_train[_labeled], 0)
    # validation, unlabelled
    X_val, X_unlab, y_val, y_unlab = train_test_split(X_remain, y_remain, train_size=val_missing)
    X_unlab, y_unlab = X_unlab[:U], y_unlab[:U]
    X_val, y_val = np.append(X_val, X_train[_validation], 0), np.append(y_val, y_train[_validation], 0)

    assert len(X_lab) == L
    assert len(y_lab) == L
    assert len(X_unlab) == U
    assert len(y_unlab) == U
    assert len(X_val) == V
    assert len(y_val) == V

    return X_lab, y_lab, X_unlab, y_unlab, X_val, y_val

def run_SS(dataset, model_class, lam=1, n_layer=1, activation='sigmoid'):
    n_reps = 1
    n_node = 2000 # num of nodes in hidden layer
    w_range = [-1, 1] # range of random weights
    b_range = [-1, 1] # range of random biases

    if dataset == g50c:
        NN = 50
        # sigma = 17.5
        partition = [50, 314, 50, 136]
    elif dataset in [coil20b, coil20]:
        NN = 2
        # sigma = 0.6
        partition = [40, 1000, 40, 360]
    elif dataset in [uspstb, uspst]:
        NN = 15
        # sigma = 9.4
        partition = [50, 1409, 50, 498]
    else:
        return 0

    data, label, n_class = dataset()

    unlab_accs = []
    val_accs = []
    test_accs = []
    train_time = []
    
    data, label = shuffle(data, label)
    # Cross-validation split
    for i in range(n_reps):
        ss = ShuffleSplit(n_splits=4, test_size=partition[3], random_state=42)
        for i, ss_values in enumerate(ss.split(data, label)):
            train_index, test_index = ss_values
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            assert len(X_test) == partition[3]
            assert len(y_test) == partition[3]

            # partition data to labelled, unlabelled, validation
            X_lab, y_lab, X_unlab, y_unlab, X_val, y_val = partition_data(X_train, y_train, partition)
            # predict unlabelled dataset
            X_train, y_train = X_lab, y_lab
            t = time()
            if model_class in [ELM, RVFL, DeepRVFL, EnsembleDeepRVFL]:
                model = model_class(n_node, lam, w_range, b_range, n_layer, activation=activation)
            elif model_class in [LapELM, LapRVFL, LapDeepRVFL, LapEnsembleDeepRVFL]:
                X_train, y_train = np.vstack([X_lab, X_unlab]), np.append(y_lab, [False]*len(y_unlab), 0)
                L = laplacian(X_lab, X_unlab, NN, sigma=2)
                model = model_class(n_node, lam, w_range, b_range, NN, L, n_layer, activation=activation)
            # elif model_class in [BRVFL, BDeepRVFL, BEnsembleDeepRVFL]:
            #     model = model_class(n_node, lam, w_range, b_range, n_layer, tol=10**(-3), activation=activation)
            model.train(X_train, y_train, n_class)
            train_time.append(time() - t)
            unlab_acc, unlab_pred = model.eval(X_unlab, y_unlab)
            # print('Unlabelled Accuracy: ', unlab_acc)
            unlab_accs.append(unlab_acc)
            # retrain on y_lab and unlab prediction, eval on val set
            new_X_train, new_y_train = np.vstack([X_lab, X_unlab]), np.append(y_lab, unlab_pred, 0)
            if model_class in [RVFL, DeepRVFL, EnsembleDeepRVFL]:
                model = model_class(n_node, lam, w_range, b_range, n_layer, activation=activation)
            elif model_class in [LapRVFL, LapDeepRVFL, LapEnsembleDeepRVFL]:
                model = model_class(n_node, lam, w_range, b_range, NN, L, n_layer, activation=activation)
            # elif model_class in [BRVFL, BDeepRVFL, BEnsembleDeepRVFL]:
            #     model = model_class(n_node, lam, w_range, b_range, n_layer, tol=10**(-5), activation=activation)
            t = time()
            model.train(new_X_train, new_y_train, n_class)
            train_time.append(time() - t)
            val_acc, _ = model.eval(X_val, y_val)
            val_accs.append(val_acc)

            test_acc, _ = model.eval(X_test, y_test)
            # print('Test accuracy: ', test_acc)val_acc
            test_accs.append(test_acc)
        break

    # find average acc
    mean_unlab_acc = 1 - np.mean(unlab_accs), np.std(unlab_accs)
    mean_val_acc = 1 - np.mean(val_accs), np.std(val_accs)
    mean_test_acc = 1 - np.mean(test_accs), np.std(test_accs)
    mean_train_time = np.mean(train_time)
    # print('Validation Accuracy', mean_val_acc)
    # print('Test accuracy: ', mean_test_acc)

    return mean_unlab_acc*100, mean_val_acc*100, mean_test_acc*100, mean_train_time

def run_all(dataset):
    run_ELM(dataset)
    run_RVFL(dataset)
    run_dRVFL(dataset)
    run_edRVFL(dataset)
    run_LapELM(dataset)
    run_LapRVFL(dataset)
    run_LapdRVFL(dataset)
    run_LapedRVFL(dataset)

def run_ELM(dataset):
    print('running ELM...')
    model_class = ELM
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=1, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_RVFL(dataset):
    print('running RVFL...')
    model_class = RVFL
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=1, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_dRVFL(dataset):
    print('running Deep RVFL...')
    model_class = DeepRVFL
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=10, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_edRVFL(dataset):
    print('running Ensemble Deep RVFL...')
    model_class = EnsembleDeepRVFL
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=10, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_LapELM(dataset):
    print('running Laplacian ELM...')
    model_class = LapELM
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lap_lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=1, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lap_lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_LapRVFL(dataset):
    print('running Laplacian RVFL...')
    model_class = LapRVFL
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lap_lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=1, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lap_lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_LapdRVFL(dataset):
    print('running Laplacian Deep RVFL...')
    model_class = LapDeepRVFL
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lap_lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=10, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lap_lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

def run_LapedRVFL(dataset):
    print('running Laplacian Ensemble Deep RVFL...')
    model_class = LapEnsembleDeepRVFL
    unlab = []
    val = []
    test = []
    t = []
    for i, lam in enumerate(lap_lams):
        unlab_acc, val_acc, test_acc, train_time = run_SS(dataset, model_class, lam=lam, n_layer=10, activation='sigmoid')
        unlab.append(unlab_acc)
        val.append(val_acc)
        test.append(test_acc)
        t.append(train_time)

    max_index = np.argmin(val, axis=0)[0]
    opt_lam = lap_lams[max_index]
    print('Unlab: ', unlab[max_index][0], u"\u00B1", unlab[max_index][1])
    print('Val: ', val[max_index][0], u"\u00B1", val[max_index][1])
    print('Test: ', test[max_index][0], u"\u00B1", test[max_index][1])
    print('Lambda: ', opt_lam)
    print('Train time: ', t[max_index])

# def run_BRVFL(data, label, n_class, partition):
#     print('running Bayesian RVFL...')
#     model_class = BRVFL
#     unlab_acc, val_acc, test_acc, train_time = run_SS(data, label, n_class, model_class, partition)

#     print('Unlab: ', unlab_acc[0], u"\u00B1", unlab_acc[1])
#     print('Val: ', val_acc[0], u"\u00B1", val_acc[1])
#     print('Test: ', test_acc[0], u"\u00B1", test_acc[1])
#     print('Train time: ', train_time)

# def run_BdRVFL(data, label, n_class, partition):
#     print('running Bayesian Deep RVFL...')
#     model_class = BDeepRVFL
#     unlab_acc, val_acc, test_acc, train_time = run_SS(data, label, n_class, model_class, partition, n_layer=10)

#     print('Unlab: ', unlab_acc[0], u"\u00B1", unlab_acc[1])
#     print('Val: ', val_acc[0], u"\u00B1", val_acc[1])
#     print('Test: ', test_acc[0], u"\u00B1", test_acc[1])
#     print('Train time: ', train_time)

# def run_BedRVFL(data, label, n_class, partition):
#     print('running Bayesian Ensemble Deep RVFL...')
#     model_class = BEnsembleDeepRVFL
#     unlab_acc, val_acc, test_acc, train_time = run_SS(data, label, n_class, model_class, partition, n_layer=10)

#     print('Unlab: ', unlab_acc[0], u"\u00B1", unlab_acc[1])
#     print('Val: ', val_acc[0], u"\u00B1", val_acc[1])
#     print('Test: ', test_acc[0], u"\u00B1", test_acc[1])
#     print('Train time: ', train_time)