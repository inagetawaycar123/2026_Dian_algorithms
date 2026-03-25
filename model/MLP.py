import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层到隐藏层的线性变换
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 隐藏层到输出层的线性变换

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        # 生成logits结果
        logits = self.fc2(x)

        #自己复现的softmax函数
        exp_logits = torch.exp(logits)
        sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
        softmax_output = exp_logits / sum_exp
        
        return logits, softmax_output



    