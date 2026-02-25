"""
模型定义：LSTM Conditional VAE + LSTM Classifier

架构：
  LSTMEncoder      : 双向 LSTM → (μ, log σ²)
  LSTMDecoder      : LSTM，以 z 初始化隐状态 → 重构序列
  LSTM_CVAE        : 编码器 + 解码器，标签条件通过 Embedding 注入
  LSTMClassifier   : 双向 LSTM + Attention → 二分类
  vae_loss()       : 重构损失 + β-KL 散度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 编码器 ──────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    双向 LSTM 编码器。
    输入: (B, T, input_dim)
    输出: μ (B, latent_dim), log σ² (B, latent_dim)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        # 双向 → hidden_dim * 2
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # h: (num_layers*2, B, hidden_dim)
        _, (h, _) = self.lstm(x)
        # 取最后一层前向和后向拼接
        h_last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden_dim*2)
        mu     = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


# ─── 解码器 ──────────────────────────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    """
    LSTM 解码器。
    用线性层将 z 映射为 LSTM 的初始隐状态，逐步解码生成序列。
    支持 Teacher Forcing（训练时）和自回归生成（推理时）。

    输入: z (B, latent_dim)
    输出: (B, T, output_dim)
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # z → 初始隐状态
        self.fc_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_c = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.lstm   = nn.LSTM(
            output_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, target=None):
        """
        z      : (B, latent_dim)
        target : (B, T, output_dim) — 训练时使用 Teacher Forcing
        """
        B = z.size(0)

        # 初始化 h0, c0
        h0 = self.fc_h(z).view(B, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()   # (num_layers, B, H)
        c0 = self.fc_c(z).view(B, self.num_layers, self.hidden_dim)
        c0 = c0.permute(1, 0, 2).contiguous()

        if target is not None:
            # Teacher Forcing：输入为真实序列（时间步前移一位，首步为零）
            sos = torch.zeros(B, 1, self.output_dim, device=z.device)
            inp = torch.cat([sos, target[:, :-1, :]], dim=1)  # (B, T, C)
        else:
            # 自回归生成（推理）：全零初始输入
            inp = torch.zeros(B, self.seq_len, self.output_dim, device=z.device)

        out, _ = self.lstm(inp, (h0, c0))
        return self.fc_out(out)   # (B, T, output_dim)


# ─── Conditional VAE ─────────────────────────────────────────────────────────

class LSTM_CVAE(nn.Module):
    """
    条件 LSTM-VAE (Conditional LSTM-VAE)。

    条件注入方式：
      编码时：将标签 Embedding 拼接到输入序列的每个时间步
      解码时：将标签 Embedding 拼接到隐向量 z

    损失 = MSE 重构损失 + β * KL 散度（支持 KL 退火）
    """

    def __init__(self,
                 input_dim=6,
                 hidden_dim=64,
                 latent_dim=16,
                 num_layers=2,
                 seq_len=256,
                 num_classes=2,
                 label_emb_dim=4,
                 dropout=0.2):
        super().__init__()
        self.input_dim    = input_dim
        self.latent_dim   = latent_dim
        self.seq_len      = seq_len
        self.label_emb_dim = label_emb_dim

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        enc_input_dim  = input_dim + label_emb_dim
        dec_latent_dim = latent_dim + label_emb_dim

        self.encoder = LSTMEncoder(enc_input_dim, hidden_dim, latent_dim,
                                   num_layers, dropout)
        self.decoder = LSTMDecoder(dec_latent_dim, hidden_dim, input_dim,
                                   seq_len, num_layers, dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # 推理时直接用均值

    def forward(self, x, label, teacher_forcing=True):
        """
        x     : (B, T, input_dim)
        label : (B,)  LongTensor
        Returns: recon (B,T,C), mu (B,latent), logvar (B,latent)
        """
        B = x.size(0)

        # 标签嵌入扩展到每个时间步
        emb     = self.label_emb(label)                               # (B, E)
        emb_seq = emb.unsqueeze(1).expand(-1, self.seq_len, -1)       # (B, T, E)

        enc_inp  = torch.cat([x, emb_seq], dim=-1)                    # (B, T, C+E)
        mu, logvar = self.encoder(enc_inp)
        z        = self.reparameterize(mu, logvar)

        z_cond   = torch.cat([z, emb], dim=-1)                        # (B, latent+E)
        recon    = self.decoder(z_cond, x if teacher_forcing else None)

        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, label, device, n_samples=1):
        """
        从指定类别生成 n_samples 条合成序列。
        返回归一化空间中的序列 (n_samples, T, C)。
        """
        self.eval()
        label_t = torch.full((n_samples,), label, dtype=torch.long, device=device)
        emb     = self.label_emb(label_t)
        z       = torch.randn(n_samples, self.latent_dim, device=device)
        z_cond  = torch.cat([z, emb], dim=-1)
        return self.decoder(z_cond, target=None)   # (n_samples, T, C)

    @torch.no_grad()
    def encode(self, x, label):
        """提取隐均值向量（用于 t-SNE 可视化）。"""
        self.eval()
        B = x.size(0)
        emb     = self.label_emb(label)
        emb_seq = emb.unsqueeze(1).expand(-1, self.seq_len, -1)
        enc_inp = torch.cat([x, emb_seq], dim=-1)
        mu, _   = self.encoder(enc_inp)
        return mu


# ─── 分类器 ──────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    双向 LSTM + Temporal Attention 分类器。
    输出: logit (B,)，配合 BCEWithLogitsLoss 使用。
    """

    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """x: (B, T, input_dim) → logit (B,)"""
        out, _ = self.lstm(x)                          # (B, T, H*2)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = (attn_w * out).sum(dim=1)            # (B, H*2)
        return self.head(context).squeeze(-1)           # (B,)


# ─── 损失函数 ─────────────────────────────────────────────────────────────────

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    CVAE 损失函数：
      L = MSE(recon, x) + β * KL(q || p)
      KL = -0.5 * mean(1 + logvar - μ² - exp(logvar))

    beta < 1 : 强调重构质量（训练初期）
    beta = 1 : 标准 VAE
    beta > 1 : β-VAE，强制更结构化的隐空间
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
