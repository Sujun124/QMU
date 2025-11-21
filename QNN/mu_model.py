import pennylane as qml
import QCNN_circuit
import unitary
import embedding
import numpy as np
import data
import Training_mu
import Training
from Benchmarking import Encoding_to_Embedding, accuracy_test
import pickle
import su_20250221
import tensorflow as tf
from sklearn.decomposition import PCA
import os
import random

dev = qml.device('default.qubit', wires=8)


def get_dataset(tag_train):
    pass

    if tag_train:
        [X_train, X_test, Y_train, Y_test] = su_20250221.load_pkl('dataset/dataset.plk')
        X_test, Y_test = X_test[0:100], Y_test[0:100]

        X_train_R, Y_train_R = X_train[Y_train != 4], Y_train[Y_train != 4]
        X_test_R, Y_test_R = X_test[Y_test != 4], Y_test[Y_test != 4]

        X_train_U, Y_train_U = X_train[Y_train == 4], Y_train[Y_train == 4]
        X_test_U, Y_test_U = X_test[Y_test == 4], Y_test[Y_test == 4]

        return [X_train_U, X_test_U, Y_train_U, Y_test_U], [X_train_R, X_test_R, Y_train_R, Y_test_R]


    else:
        X_test, Y_test = su_20250221.load_pkl(f'dataset/dataset_all_label.plk')
        X_test_U = X_test.pop(4)
        Y_test_U = Y_test.pop(4)

        return X_test_U, Y_test_U, X_test, Y_test


def process(Unitary,Encodings,args):
    U = Unitary
    U_params = U_num_param

    tag_train = True
    if tag_train:
        [X_train, X_test, Y_train, Y_test] = su_20250221.load_pkl('dataset/dataset_all.plk')
        args.total_params = 8 * 3
        [history, trained_params, args, indices] = su_20250221.load_pkl(f'20250309/trained_params_10_0.05_32_0308.pkl')

        # indices_train = np.random.choice(range(len(X_train)), size=args.n_train, replace=False)
        # indices_test = np.random.choice(range(len(X_test)), size=args.n_test, replace=False)
        X_train, X_test, Y_train, Y_test = X_train[indices[0]], X_test[indices[1]], Y_train[indices[0]], Y_test[
            indices[1]]

        X_train_R, Y_train_R = X_train[Y_train != 4], Y_train[Y_train != 4]
        X_test_R, Y_test_R = X_test[Y_test != 4], Y_test[Y_test != 4]

        X_train_U, Y_train_U = X_train[Y_train == 4], Y_train[Y_train == 4]
        X_test_U, Y_test_U = X_test[Y_test == 4], Y_test[Y_test == 4]
        args.noise = False

        history, trained_params = Training_mu.circuit_training(args,
                                         [X_train_U, X_train_R], [Y_train_U, Y_train_R],
                                         trained_params,
                                         U, U_params,
                                         Encodings, circuit,
                                         cost_fn='multi_cross_entropy',
                                         test=[[X_test_U, X_test_R], [Y_test_U, Y_test_R]])

        # draw and save
        lists, name1, name2, save_path = history, ['iter', 'acc', f'acc{args.name}'], ['loss', 'acc'], 'Result/'
        su_20250221.draw_sub_data_n(lists, name1, name2, save_path)
        name = f'result/trained_params_{args.name}_mu_g_0921.pkl'
        f = open(name, 'wb')
        pickle.dump([history, trained_params,args], file=f)
        f.close()

    tag_test = False
    if tag_test:

        X_test, Y_test = su_20250221.load_pkl(f'dataset/dataset_all_label.plk')

        [history, trained_params,args] = su_20250221.load_pkl(f'result/trained_params_10_0.05_32_0308_epoch0_mu_g_UR5.pkl')

        trained_params = su_20250221.load_pkl(f'result/trained_params_mu_g_for_compare_train_size_epoch0.pkl')

        args.total_params = 24
        args.noise = False
        accuracy_list = []
        for i in range(len(X_test)):
            accuracy = Training_mu.cost(args, trained_params, X_test[i], Y_test[i], 'Hardware_efficiency', '', Encodings, circuit,
                                        cost_fn='multi_cross_entropy',
                                        tag_test=True)

            accuracy_list.append(accuracy)

        # index = accuracy_list.index(max(accuracy_list))
        # best_trained_params_list.append(trained_params_list[index])
        for i in range(len(accuracy_list)):
            print(f'label: {i}, accuracy：{accuracy_list[i].item()}')


if __name__ == "__main__":
    time_begin, time_save = su_20250221.read_curr_time()
    print(time_save)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=2, help="random seed")
    parser.add_argument('-save_path', type=str, default='result', help="save path")
    parser.add_argument('-time_save', type=str, default='XX', help="save path")
    parser.add_argument('-name', type=str, default='XX', help="name")

    parser.add_argument('-n_qubit', type=int, default=8, help="n_qubit")
    parser.add_argument('-n_layers', type=int, default=5, help="n_layers")

    parser.add_argument('-n_epochs', type=int, default=10, help="n_epochs")
    parser.add_argument('-batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('-learning_rate', type=float, default=0.05, help="learning_rate")

    parser.add_argument('-n_train', type=int, default=1000, help="n_train")
    parser.add_argument('-n_test', type=int, default=500, help="n_test")
    parser.add_argument('-noise', type=bool, default=False, help="n_test")


    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    su_20250221.creat_file(args.save_path)
    args.time_save = time_save

    args.name = f'{args.n_layers}_{args.learning_rate}_{args.batch_size}_0611'

    dataset = 'mnist'
    classes = [0, 1]

    circuit = 'Hardware_efficiency'
    Unitary = 'Hardware_efficiency'
    U_num_param = 8 * 3

    # U_num_param = 15
    # circuit = 'QCNN'
    # Unitary = 'U_SU4'

    Encodings = 'Amplitude'
    print('name', args.name)
    process(Unitary,Encodings,args)

    time_end, _ = su_20250221.read_curr_time()
    print('time', time_end - time_begin)

    print("Hardware_efficiency")
