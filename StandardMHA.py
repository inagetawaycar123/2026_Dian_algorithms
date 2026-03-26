import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除"

        self.embed_dim = embed_dim      # 输入嵌入的维度
        self.num_heads = num_heads      # 注意力头的数量
        self.head_dim = embed_dim // num_heads      # 每个头的维度
        self.scale = self.head_dim ** -0.5      # 缩放因子1/√d_k
        self.dropout = nn.Dropout(dropout)

        # 定义线性层用于生成查询、键、值
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 拼接后的输出线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影得到 Q, K, V  [batch, seq_len, embed_dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 分割成多个头 [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 缩放点积注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]

        # 4. 掩码（可选）：mask 为 True 的位置置为 -inf，softmax 后权重为 0
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == False, -1e9)

        # 5. Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. 由注意力权重计算出结果
        attn_output = torch.matmul(attn_weights, V)

        # 7. 拼接多个头的输出 [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 8. 最后线性投影
        output = self.out_proj(attn_output)

        return output, attn_weights
    


# ===================== 前向验证 =====================
# 1. 设置超参数
batch_size = 3    # 批次大小
seq_len = 12      # 序列长度（句子长度）
hidden_dim = 512  # 隐藏层维度 = embed_dim
num_heads = 8     # 注意力头数

# 2. 随机生成输入张量：形状 (batch_size, seq_len, hidden_dim)
x = torch.randn(batch_size, seq_len, hidden_dim)

# 3. 初始化 MHA 层
mha = StandardMHA(embed_dim=hidden_dim, num_heads=num_heads)

# 4. 前向传播
output, attn_weights = mha(x)

# 5. 打印输入 & 输出形状
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")

# 6. 严格验证：输入输出形状是否完全一致
assert x.shape == output.shape, "ERROR：输入输出形状不一致！"
print("\n✅ 验证成功：输入输出形状完全一致！")
