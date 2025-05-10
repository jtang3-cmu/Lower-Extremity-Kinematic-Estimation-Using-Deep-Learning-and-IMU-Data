import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# RoPE 輔助函式
# -----------------------
def get_rotary_embedding(seq_len, dim, device):
    # 計算偶數位置的反頻率
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # shape: [seq_len, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)         # shape: [seq_len, dim]
    # 新增 singleton 維度，讓 seq_len 位置在 dim 2
    cos_emb = emb.cos()[None, None, :, :]    # shape: [1, 1, seq_len, dim]
    sin_emb = emb.sin()[None, None, :, :]      # shape: [1, 1, seq_len, dim]
    return cos_emb, sin_emb

def rotate_every_two(x):
    # x: [batch, num_heads, seq_len, head_dim]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # 旋轉: (-x2, x1)
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return x_rotated.flatten(-2)  # 恢復 head_dim

def apply_rope(x, cos, sin):
    # x: [batch, num_heads, seq_len, head_dim]
    return x * cos + rotate_every_two(x) * sin

# -----------------------
# 自訂 Multi-Head Attention (RoPE)
# -----------------------
class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各自 shape: [batch, num_heads, seq_len, head_dim]

        # 計算 RoPE embedding
        device = x.device
        cos_emb, sin_emb = get_rotary_embedding(seq_len, self.head_dim, device)
        q = apply_rope(q, cos_emb, sin_emb)
        k = apply_rope(k, cos_emb, sin_emb)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1,2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        return output

# -----------------------
# Transformer Encoder Layer (RoPE)
# -----------------------
class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=256, dropout=0.1, max_seq_len=100, **kwargs):
        super().__init__()
        self.self_attn = MultiHeadAttentionRoPE(d_model, num_heads, max_seq_len)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # 自注意力區塊：殘差連接與正規化
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feed-forward 區塊：殘差連接與正規化
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# -----------------------
# Transformer 模型 (RoPE)
# -----------------------
class TransformerModelRoPE(nn.Module):
    def __init__(self, inp_size, d_model, num_heads, num_layers, outp_size, max_seq_len, dim_feedforward=256, dropout=0.1, **kwargs):
        super().__init__()
        # 將輸入投影到模型維度
        self.input_linear = nn.Linear(inp_size, d_model)
        # 堆疊多層 Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerRoPE(d_model, num_heads, dim_feedforward, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        # 最後的輸出投影：對每個時間點進行預測
        self.output_linear = nn.Linear(d_model, outp_size)

    def forward(self, x):
        # x: [batch, seq_len, inp_size]
        x = self.input_linear(x)
        for layer in self.encoder_layers:
            x = layer(x)
        out = self.output_linear(x)
        return out

