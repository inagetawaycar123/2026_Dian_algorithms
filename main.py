import os
import torch
import torch.nn as nn
from model.MLP import MLP
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def main():
    # 超参数配置
    input_dim = 4
    hidden_dim = 16
    output_dim = 3
    epochs = 50

    print("脚本开始运行，准备构建模型和加载数据...")

    # 模型、损失函数和优化器的定义
    model = MLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 加载数据
    data_path = os.path.join("data", "iris.data")
    print("当前数据路径：", data_path)
    df = pd.read_csv(data_path, header=None)

    # 定义x，y
    x_data = df.iloc[:, :4].values
    y_data = df.iloc[:, 4].values

    # 数字编码
    encoder = LabelEncoder()
    y_data = encoder.fit_transform(y_data)

    # 转化为pytorch向量
    x = torch.tensor(x_data, dtype=torch.float32)
    labels = torch.tensor(y_data, dtype=torch.long)

    # 训练循环
    for epoch in range(epochs):
        model.train()

        logits, softmax_output = model(x)
        loss = criterion(logits, labels)

        optimizer.zero_grad()  # 清空上一轮的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("训练结束。")


if __name__ == "__main__":
    main()

