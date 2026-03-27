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

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: torch.Tensor = None,
        past_key_values: tuple[torch.Tensor, torch.Tensor] = None
    ):
        
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影得到 Q, K, V  [batch, seq_len, embed_dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 分割成多个头 [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            K = torch.cat([past_k, K], dim=-2)
            V = torch.cat([past_v, V], dim=-2)

        new_kv = (K, V)

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

        return output, new_kv

def main():
     # 超参数
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    initial_len = 10   # 初始序列长度
    generate_steps = 5 # 生成 5 个新 token

    # 初始化模型
    model = StandardMHA(embed_dim=hidden_dim, num_heads=num_heads)

    print("=" * 70)
    print(f"📌 第一步：输入初始序列，长度 = {initial_len}")
    # 初始输入 (B, 10, C)
    x_init = torch.randn(batch_size, initial_len, hidden_dim)
    out, past_kv = model(x_init)

    # 查看初始缓存
    k_cache, v_cache = past_kv
    print(f"初始 K 形状: {k_cache.shape}")
    print(f"初始 V 形状: {v_cache.shape}")

    print("=" * 70)
    print("📌 开始自回归生成（每次只输入 1 个 token）\n")

    cache_length = []

    # 循环生成 5 次
    for i in range(generate_steps):
        # 每次只输入 1 个新 token
        x_new = torch.randn(batch_size, 1, hidden_dim)

        # 传入缓存，前向
        out, past_kv = model(x_new, past_key_values=past_kv)

        # 取出当前缓存
        k, v = past_kv
        current_seq_len = k.shape[-2]
        cache_length.append(current_seq_len)

        print(f"✅ 第 {i+1} 步")
        print(f"   输入 Q 序列长度: {x_new.shape[1]} (永远是 1)")
        print(f"   KV 缓存序列长度: {current_seq_len}")
        print(f"   K 形状: {k.shape}\n")

    expected_lengths = [initial_len + i + 1 for i in range(generate_steps)]
    assert cache_length == expected_lengths, f"ERROR：KV Cache 长度不正确！期望 {expected_lengths}，但得到 {cache_length}"
    
    print("=" * 70)
    print("🎉 最终长度：10 + 5 = 15，KV Cache 验证成功！")

if __name__ == "__main__":
    main()
