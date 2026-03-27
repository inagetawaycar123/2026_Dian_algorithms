import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim % self.num_kv_heads == 0, "每个头的维度必须能被 KV 头数整除"

        # 每组内有多少个 Q 共享一个 KV
        self.n_group = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_key_values=None):
        B, T, C = x.shape

        # 1. 线性投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 分割成多个头
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, T, head_dim]
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, T, head_dim]

        # 3. KV Cache 的实现
        if past_key_values is not None:
            past_k, past_v = past_key_values
            K = torch.cat([past_k, K], dim=-2)  # 拼接历史 K
            V = torch.cat([past_v, V], dim=-2)  # 拼接历史 V
        new_kv = (K, V)
        T_k = K.size(-2)  # 当前 K 的序列长度

        # 4. KV 形状 [B, H_kv, T, D] → 扩展为 [B, H, T, D] 匹配 Q
        K = K.repeat_interleave(self.n_group, dim=1)
        V = V.repeat_interleave(self.n_group, dim=1)

        # 5. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # 6. 拼接多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)

        return output, new_kv
    

def main():
    batch_size = 2
    seq_len = 10
    hidden_dim = 512
    num_heads = 8
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # 保存所有结果
    all_pass = True

    # ====================== 测试 1: MHA ======================
    print("=" * 70)
    print("🧪 测试 MHA (num_kv_heads = 8)")
    mha = GroupedQueryAttention(embed_dim=hidden_dim, num_heads=num_heads, num_kv_heads=8)
    out, (k, v) = mha(x)

    # 真正判断
    is_mha_correct = (
        k.shape[1] == 8
        and out.shape == x.shape
    )
    print(f"K 形状: {k.shape}")
    print(f"输出形状匹配: {out.shape == x.shape}")
    print(f"✅ MHA 正确: {is_mha_correct}")
    if not is_mha_correct:
        all_pass = False

    # ====================== 测试 2: GQA ======================
    print("=" * 70)
    print("🧪 测试 GQA (num_kv_heads = 2)")
    gqa = GroupedQueryAttention(embed_dim=hidden_dim, num_heads=num_heads, num_kv_heads=2)
    out, (k, v) = gqa(x)

    is_gqa_correct = (
        k.shape[1] == 2
        and out.shape == x.shape
    )
    print(f"K 形状: {k.shape}")
    print(f"输出形状匹配: {out.shape == x.shape}")
    print(f"✅ GQA 正确: {is_gqa_correct}")
    if not is_gqa_correct:
        all_pass = False

    # ====================== 测试 3: MQA ======================
    print("=" * 70)
    print("🧪 测试 MQA (num_kv_heads = 1)")
    mqa = GroupedQueryAttention(embed_dim=hidden_dim, num_heads=num_heads, num_kv_heads=1)
    out, (k, v) = mqa(x)

    is_mqa_correct = (
        k.shape[1] == 1
        and out.shape == x.shape
    )
    print(f"K 形状: {k.shape}")
    print(f"输出形状匹配: {out.shape == x.shape}")
    print(f"✅ MQA 正确: {is_mqa_correct}")
    if not is_mqa_correct:
        all_pass = False

    # ====================== 最终结论 ======================
    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 最终结论：GroupedQueryAttention 成功统一 MHA / GQA / MQA！")
    else:
        print("❌ 验证失败：未能正确统一三种注意力机制")
    print("=" * 70)

if __name__ == "__main__":
    main()