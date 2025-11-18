import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
import random
import pickle

qubit_num = n_qubits = 8
layer_n = 1
# 定义量子设备
dev = qml.device('default.qubit', wires=qubit_num)


# 计算测试集的准确率
def calculate_accuracy(model, testloader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测标签
            total += labels.size(0)  # 样本数量
            correct += (predicted == labels).sum().item()  # 计算正确预测的数量

    accuracy = correct / total
    return accuracy

# 定义量子神经网络
@qml.qnode(dev,interface="torch")
def qnode(inputs,QNN_param):

    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

    for k in range(args.n_layers):
        for j in range(n_qubits):
            qml.U3(*QNN_param[k][j], wires=[j])

        for j in range(n_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class ConvQNN(nn.Module):
    def __init__(self,args):
        super(ConvQNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.fc1 = nn.Linear(980, 256)  # 全连接层16:3136 5:980
        self.fc2 = nn.Linear(8, 10)  # 输出层
        self.softmax = nn.LogSoftmax(dim=1)
        weight_shapes = {"QNN_param": (args.n_layers, args.n_qubit, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        with torch.no_grad():
            self.qlayer.qnode_weights["QNN_param"].uniform_(-np.pi, np.pi)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积和池化
        x = x.view(x.shape[0], -1)  # 展平
        x = self.fc1(x)  # 全连接层
        x = F.leaky_relu(x, negative_slope=0.01)
        x = torch.stack([self.qlayer(xi / (torch.norm(xi) + 1e-8)) for xi in x])
        x = self.fc2(x)  # 输出层
        return x
def process(args):


    # 设置变换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 抽样索引
    train_indices_all = random.sample(range(len(trainset)), args.n_train)
    test_indices_all = random.sample(range(len(testset)), args.n_test)

    # 按 label 分成 R（非4）和 U（=4）
    def split_indices_by_label(dataset, indices, target_label=4):
        r_indices, u_indices = [], []
        for idx in indices:
            _, label = dataset[idx]
            if label == target_label:
                u_indices.append(idx)
            else:
                r_indices.append(idx)
        return r_indices, u_indices

    train_r_idx, train_u_idx = split_indices_by_label(trainset, train_indices_all)
    test_r_idx, test_u_idx = split_indices_by_label(testset, test_indices_all)

    # 构造子集和数据加载器
    train_r_set = Subset(trainset, train_r_idx)
    test_r_set = Subset(testset, test_r_idx)
    test_u_set = Subset(testset, test_u_idx)

    trainloader = DataLoader(train_r_set, batch_size=args.batch_size, shuffle=True)
    test_r_loader = DataLoader(test_r_set, batch_size=args.batch_size, shuffle=False)
    test_u_loader = DataLoader(test_u_set, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = ConvQNN(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    lists = [[], [], [], []]  # [loss, acc_R, acc_U]

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        correct, total = 0, 0

        model.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        acc_r = calculate_accuracy(model, test_r_loader)
        acc_u = calculate_accuracy(model, test_u_loader)

        lists[0].append(running_loss / len(trainloader))  # Loss
        lists[1].append(train_acc)  # Accuracy on R
        lists[2].append(acc_r)  # Accuracy on R
        lists[3].append(acc_u)  # Accuracy on U

        print(f"Epoch {epoch + 1}, Loss: {lists[0][-1]:.2f}, train_acc: {train_acc:.2f}, Acc_R: {acc_r:.2f}, Acc_U: {acc_u:.2f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "args": args,
        "lists": lists
    }, f"model{args.name}.pth")

    # 保存数据
    save_name = f'{args.save_path}/trained_params_{args.name}.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump([lists, args], f)




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=0, help="random seed")
    parser.add_argument('-save_path', type=str, default='result/', help="save path")
    parser.add_argument('-time_save', type=str, default='XX', help="save path")
    parser.add_argument('-name', type=str, default='XX', help="name")

    parser.add_argument('-n_qubit', type=int, default=8, help="n_qubit")
    parser.add_argument('-n_layers', type=int, default=1, help="n_layers")

    parser.add_argument('-n_epochs', type=int, default=10, help="n_epochs")
    parser.add_argument('-batch_size', type=int, default=8, help="batch_size")
    parser.add_argument('-learning_rate', type=float, default=0.1, help="learning_rate")

    parser.add_argument('-n_train', type=int, default=1000, help="n_train")
    parser.add_argument('-n_test', type=int, default=500, help="n_test")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.name = f'0324_target_{args.n_layers}_{args.learning_rate}_{args.batch_size}'

    process(args)




