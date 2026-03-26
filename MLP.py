import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层的线性变换
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层的线性变换

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        # 生成 logits 结果
        logits = self.fc2(x)

        # 自己复现的 softmax 函数
        exp_logits = torch.exp(logits)
        sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
        softmax_output = exp_logits / sum_exp

        return logits, softmax_output


def main():
    # 超参数配置
    input_dim = 4
    hidden_dim = 16
    output_dim = 3
    epochs = 300

    print("脚本开始运行，准备构建模型和加载数据...")

    # 模型、损失函数和优化器的定义
    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 加载数据
    data_path = os.path.join("data", "iris.data")
    print("当前数据路径：", data_path)
    df = pd.read_csv(data_path, header=None)

    # 定义 x, y
    x_data = df.iloc[:, :4].values
    y_data = df.iloc[:, 4].values

    # 数字编码
    encoder = LabelEncoder()
    y_data = encoder.fit_transform(y_data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42  # 20% 用于测试
    )

    # 转 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 训练循环
    for epoch in range(epochs):
        model.train()

        logits, softmax_output = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()  # 清空上一轮的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("训练结束。")

    print("\n========== 测试集结果 ==========")
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits, softmax_output = model(X_test)
        _, predicted = torch.max(softmax_output, 1)  # 预测类别
        correct = (predicted == y_test).sum().item()
        acc = correct / len(y_test)

    print(f"测试集正确数量: {correct}/{len(y_test)}")
    print(f"测试集准确率: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
