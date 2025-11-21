# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp


def compute_gradients(args, params, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn,
                      tag_cla10=True):
    """
    使用pennylane内置的grad函数计算梯度。
    """
    grad_fn = qml.grad(cost, argnum=1)
    gradients = grad_fn(args, params, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn,
                        tag_cla10, False)
    return gradients

def one_hot_encode(y, num_classes):
    """
    将标签转换为 one-hot 形式
    :param y: 原始类别标签列表 (如 [1,6,4])
    :param num_classes: 类别总数 (如 10)
    :return: one-hot 编码矩阵
    """
    one_hot = anp.zeros((len(y), num_classes))  # 创建 (样本数, 类别数) 的零矩阵
    one_hot[anp.arange(len(y)), y] = 1  # 将对应类别位置设为 1
    return one_hot

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss


def cross_entropy_multi(y, predictions):
    loss = 0
    for i in range(len(y)):
        for l, p in zip(y[i], predictions[i]):
            c_entropy = l * (anp.log(p)) + (1 - l) * anp.log(1 - p)
            loss = loss + c_entropy
    return -1 * loss

def cal_accuracy(Y, predictions):
    predicted_classes = np.argmax(predictions, axis=1)  # 取最大值索引

    # 计算正确预测的个数
    true_classes = np.argmax(Y, axis=1)
    correct_predictions = np.sum(predicted_classes == true_classes)
    accuracy = correct_predictions / len(Y)
    return accuracy

import autograd.numpy as anp

def softmax(x):
    """数值稳定的 Softmax 实现，确保总和等于 1"""
    exp_x = anp.exp(x - anp.max(x))  # 防止溢出
    return exp_x / anp.sum(exp_x)  # 归一化，确保总和为 1




def cost(args,params, X, Y, U, U_params, embedding_type, circuit, cost_fn,tag_cla10=True,tag_test=False):
    if tag_cla10:

        params_dense_b = params[-10::]
        params_dense_w = params[-90:-10]
        params_circuit = params[0:-90]

        # qnn
        predictions = [QCNN_circuit.QCNN(args,x, params_circuit, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
        weights = anp.reshape(params_dense_w[:80], (8, 10)) #参数改维度，是为了后续方便矩阵相乘
        out_list = []
        for i in range(len(predictions)): # batch
            out = anp.dot(anp.array(predictions[i]), weights) # 全连接层FC： 权重相乘
            out = anp.add(out, params_dense_b)  # 全连接层FC：bisi 相加
            out = softmax(out) # 激活层
            out_list.append(out)
        # 攻击QNN隐私漏洞用的，不用管
        if tag_test=="MIA":
            return (out_list,[np.max(out_list[i]) for i in range(len(out_list))],
                    [int(np.argmax(out_list[i])==Y[i]) for i in range(len(out_list))],
                    [cross_entropy_multi(one_hot_encode([Y[i]], 10), [out_list[i]]) for i in range(len(Y))])
        # compute acc
        elif tag_test=="acc":
            Y = one_hot_encode(Y, 10)
            acc = cal_accuracy(Y, out_list)
            return acc
        # compute loss
        else:
            Y = list(one_hot_encode(Y, 10))
            Y = np.array(Y)
            out_list = np.array(out_list)
            loss = [cross_entropy_multi([Y[i]], [out_list[i]]).numpy() for i in range(len(out_list))]
            return loss

    else: # 不用管
        if circuit == 'QCNN':
            predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
        elif circuit == 'Hierarchical':
            predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
        if cost_fn == 'mse':
            loss = square_loss(Y, predictions)
        elif cost_fn == 'cross_entropy':
            loss = cross_entropy(Y, predictions)

        return loss/len(Y)

# Circuit training parameters

def circuit_training(args,X_train, Y_train, U, U_params, embedding_type, circuit,
                     cost_fn='cross_entropy', cla10=False,test=False):

    # 选择参数
    if circuit == 'QCNN':
        if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
            total_params = U_params * 3
        else:
            total_params = U_params * 3 + 2 * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    elif circuit == 'Hardware_efficiency':
        total_params = args.n_qubit*3

    # 后处理层
    if cla10:
        total_params_w = 8*10    # w
        total_params_b = 1*10    # b

    args.total_params = total_params
    n_params = total_params*args.n_layers + total_params_w + total_params_b

    params = np.random.randn(n_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=args.learning_rate)
    history = [[],[]]

    for it in range(args.n_epochs):
        loss_batch = []

        train_all = args.train_all
        if train_all:
            indices = np.random.permutation(len(X_train))
            X_train = np.array(X_train)[indices]
            Y_train = np.array(Y_train)[indices]

            for i in range(0, len(X_train), args.batch_size):

                X_batch = X_train[i: i + args.batch_size]
                Y_batch = Y_train[i: i + args.batch_size]
                assert isinstance(params, object)
                # gradients = compute_gradients(args, params, X_batch, Y_batch, U, U_params, embedding_type, circuit,
                #                               cost_fn)
                import su_20250221
                time_record1, time_save = su_20250221.read_curr_time()

                # cost：QNN的前向传播（执行一次QNN）
                # step_and_cost：更新
                params, cost_new = opt.step_and_cost(lambda v: cost(args,v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn,tag_cla10=True,tag_test=False),
                                                              params)
                loss_batch.append(cost_new)
                time_record2, time_save = su_20250221.read_curr_time()
                print(time_record2-time_record1)
                exit()
        else:
            indices = np.random.choice(len(X_train), args.batch_size, replace=False)  # 随机抽样
            X_batch = X_train[indices]
            Y_batch = Y_train[indices]

            params, cost_new = opt.step_and_cost(
                lambda v: cost(args, v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn, tag_cla10=True,
                               tag_test=False),
                params)

            loss_batch.append(cost_new)

        history[0].append(np.average(loss_batch))

        if True:
            if test:
                acc = cost(args,params, test[0], test[1], U, U_params, embedding_type, circuit, cost_fn,tag_cla10=True,tag_test=True)
                history[1].append(acc)
                print("iteration: ", it+1,
                      " cost: ", history[0][-1],
                      " acc: ", history[1][-1])
            else:
                print("iteration: ", it+1, " cost: ", history[0][-1])

        if (it+1) % 20 == 0:
            import pickle
            name = f'result/trained_params_{(it+1)}.pkl'
            f = open(name, 'wb')
            pickle.dump([history, params, args], file=f)
            f.close()

    return history, params


