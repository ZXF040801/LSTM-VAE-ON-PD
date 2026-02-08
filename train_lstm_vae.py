"""
Conv1D-VAE 训练脚本 (v3 - 全卷积架构)
======================================
核心改进:
1. 全卷积编解码器（替代LSTM）- Conv1D在时序重构方面远优于LSTM
2. 空间潜在空间: z形状=(B,64,45)，有效潜在维度=2880，远大于信号的504有效维度
3. 渐进式下采样/上采样（360→180→90→45），避免极端压缩瓶颈
4. 两阶段训练：AE预训练 → VAE微调

前置条件: 先运行修改后的 pd_preprocessing.py 重新生成数据（全局标准化）

运行: python train_lstm_vae.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    # 数据路径
    DATA_FOLDER = r'D:\Final work\DataForStudentProject-HWU\processed_data'
    OUTPUT_FOLDER = r'D:\Final work\DataForStudentProject-HWU\vae_results'

    # 模型参数
    INPUT_DIM = 12          # 输入通道数（6传感器 × 2手指 = 12）
    SEQ_LEN = 360           # 序列长度
    LATENT_CHANNELS = 64    # 潜在空间通道数
    # 经过3次stride=2下采样: 360→180→90→45
    # 有效潜在维度 = 64 × 45 = 2880 (远大于信号的504有效维度)

    # 训练参数
    BATCH_SIZE = 16
    AE_EPOCHS = 200         # 阶段1：AE预训练（需要更多轮次让卷积网络收敛）
    VAE_EPOCHS = 100        # 阶段2：VAE微调
    AE_LR = 1e-3
    VAE_LR = 3e-4
    MAX_BETA = 0.001        # KL权重极小，优先重构质量

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 卷积编码器
# ============================================================================

class ConvEncoder(nn.Module):
    """
    全卷积编码器：(B, 12, 360) → 空间潜在分布 (B, 64, 45)

    通过3次stride=2下采样逐步压缩时间维度:
    360 → 180 → 90 → 45
    """
    def __init__(self, input_dim=12, latent_channels=64, class_embed_dim=16):
        super().__init__()
        in_ch = input_dim + class_embed_dim  # 12 + 16 = 28

        self.encoder = nn.Sequential(
            # Block 1: (B, 28, 360) → (B, 64, 180)
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            # Block 2: (B, 64, 180) → (B, 128, 90)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            # Block 3: (B, 128, 90) → (B, 256, 45)
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # Refine: (B, 256, 45) → (B, 256, 45)
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # 空间潜在变量：在每个位置独立预测mu和logvar
        self.conv_mu = nn.Conv1d(256, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv1d(256, latent_channels, kernel_size=1)

    def forward(self, x_cond):
        """
        x_cond: (B, 28, 360) = input + class_embedding
        returns: mu (B, 64, 45), logvar (B, 64, 45)
        """
        h = self.encoder(x_cond)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


# ============================================================================
# 卷积解码器
# ============================================================================

class ConvDecoder(nn.Module):
    """
    全卷积解码器：空间潜在变量 (B, 64, 45) → (B, 12, 360)

    通过3次上采样+卷积逐步恢复时间维度:
    45 → 90 → 180 → 360
    """
    def __init__(self, output_dim=12, latent_channels=64, class_embed_dim=16):
        super().__init__()
        in_ch = latent_channels + class_embed_dim  # 64 + 16 = 80

        self.decoder = nn.Sequential(
            # Expand: (B, 80, 45) → (B, 256, 45)
            nn.Conv1d(in_ch, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Refine: (B, 256, 45) → (B, 256, 45)
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Block 1: Upsample 45→90 then conv
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Block 2: Upsample 90→180 then conv
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 3: Upsample 180→360 then conv
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output: (B, 64, 360) → (B, 12, 360)
            nn.Conv1d(64, output_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z_cond):
        """z_cond: (B, 80, 45), returns: (B, 12, 360)"""
        return self.decoder(z_cond)


# ============================================================================
# CVAE主模型
# ============================================================================

class ConvCVAE(nn.Module):
    """
    条件卷积VAE

    关键设计:
    - 空间潜在变量: z形状=(B, 64, 45)，不是传统的(B, latent_dim)向量
    - 这意味着有效潜在维度 = 64×45 = 2880，足以编码信号中的504个有效自由度
    - 条件信息(class)通过通道拼接注入
    """
    def __init__(self, input_dim=12, latent_channels=64, num_classes=2, seq_len=360):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_classes = num_classes
        self.class_embed_dim = 16
        self.spatial_len = seq_len // 8  # 360/8 = 45

        self.class_embedding = nn.Embedding(num_classes, self.class_embed_dim)
        self.encoder = ConvEncoder(input_dim, latent_channels, self.class_embed_dim)
        self.decoder = ConvDecoder(input_dim, latent_channels, self.class_embed_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, labels):
        """
        x: (B, seq_len, input_dim) — 注意输入是 (B, T, C) 格式
        labels: (B,)
        """
        batch_size = x.size(0)

        # 转换为 (B, C, T) 用于Conv1d
        x_t = x.transpose(1, 2)  # (B, 12, 360)

        # 类别条件
        class_emb = self.class_embedding(labels)  # (B, 16)

        # --- Encoder ---
        # 拼接类别到输入通道
        class_spatial_in = class_emb.unsqueeze(2).expand(-1, -1, x_t.size(2))  # (B, 16, 360)
        x_cond = torch.cat([x_t, class_spatial_in], dim=1)  # (B, 28, 360)

        mu, logvar = self.encoder(x_cond)  # (B, 64, 45), (B, 64, 45)
        z = self.reparameterize(mu, logvar)  # (B, 64, 45)

        # --- Decoder ---
        # 拼接类别到潜在变量
        class_spatial_z = class_emb.unsqueeze(2).expand(-1, -1, z.size(2))  # (B, 16, 45)
        z_cond = torch.cat([z, class_spatial_z], dim=1)  # (B, 80, 45)

        x_recon_t = self.decoder(z_cond)  # (B, 12, 360)

        # 转回 (B, T, C)
        x_recon = x_recon_t.transpose(1, 2)  # (B, 360, 12)

        return x_recon, mu, logvar, z

    def encode(self, x, labels=None):
        """编码到潜在空间"""
        if labels is None:
            labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_t = x.transpose(1, 2)
        class_emb = self.class_embedding(labels)
        class_spatial_in = class_emb.unsqueeze(2).expand(-1, -1, x_t.size(2))
        x_cond = torch.cat([x_t, class_spatial_in], dim=1)

        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate(self, labels, num_samples=None, device='cpu'):
        """从先验分布采样生成"""
        if isinstance(labels, int):
            labels = torch.tensor([labels] * num_samples).to(device)

        batch_size = labels.size(0)
        z = torch.randn(batch_size, self.latent_channels, self.spatial_len).to(device)

        class_emb = self.class_embedding(labels)
        class_spatial_z = class_emb.unsqueeze(2).expand(-1, -1, z.size(2))
        z_cond = torch.cat([z, class_spatial_z], dim=1)

        x_recon_t = self.decoder(z_cond)
        return x_recon_t.transpose(1, 2)  # (B, 360, 12)


# ============================================================================
# 损失函数
# ============================================================================

def vae_loss(x_recon, x, mu, logvar, beta=0.0):
    """
    VAE损失:
    - 重构损失: MSE sum over (T, C), mean over batch
    - KL散度: 对空间潜在变量的每个位置求和
    """
    batch_size = x.size(0)

    # 重构损失
    recon_loss = F.mse_loss(x_recon, x, reduction='none')
    recon_loss = recon_loss.sum(dim=[1, 2]).mean()

    # KL散度 (空间潜在变量: sum over channels and spatial positions)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ============================================================================
# 两阶段训练
# ============================================================================

def train_stage1_ae(model, train_loader, val_loader, config):
    """阶段1: AE预训练 (beta=0, 纯重构)"""
    print("\n" + "=" * 70)
    print("  阶段1: 自编码器预训练 (beta=0, 纯重构)")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.AE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.AE_EPOCHS, eta_min=1e-5)

    save_path = os.path.join(config.OUTPUT_FOLDER, 'cvae_ae_pretrain.pth')
    history = {'train_recon': [], 'val_recon': [], 'val_corr': []}
    best_val_recon = float('inf')

    for epoch in range(config.AE_EPOCHS):
        model.train()
        train_recon = 0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(batch_x, batch_y)
            loss, recon_loss, _ = vae_loss(x_recon, batch_x, mu, logvar, beta=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_recon += recon_loss.item()
            n_batches += 1

        train_recon /= n_batches
        scheduler.step()

        # 验证
        model.eval()
        val_recon, val_corr, n_val, n_corr = 0, 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
                x_recon, mu, logvar, _ = model(batch_x, batch_y)
                _, recon_loss, _ = vae_loss(x_recon, batch_x, mu, logvar, beta=0.0)
                val_recon += recon_loss.item()
                n_val += 1
                for i in range(batch_x.size(0)):
                    orig = batch_x[i].cpu().numpy().flatten()
                    pred = x_recon[i].cpu().numpy().flatten()
                    corr = np.corrcoef(orig, pred)[0, 1]
                    if not np.isnan(corr):
                        val_corr += corr
                        n_corr += 1

        val_recon /= max(n_val, 1)
        val_corr /= max(n_corr, 1)

        history['train_recon'].append(train_recon)
        history['val_recon'].append(val_recon)
        history['val_corr'].append(val_corr)

        if val_recon < best_val_recon:
            best_val_recon = val_recon
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{config.AE_EPOCHS}] "
                  f"Train Recon: {train_recon:.1f} | Val Recon: {val_recon:.1f} | Corr: {val_corr:.4f}")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"\n  阶段1完成! 最佳Val Recon: {best_val_recon:.1f}")
    print(f"  最佳Val Corr: {max(history['val_corr']):.4f}")
    return history


def train_stage2_vae(model, train_loader, val_loader, config):
    """阶段2: VAE微调 (缓慢引入KL)"""
    print("\n" + "=" * 70)
    print(f"  阶段2: VAE微调 (max_beta={config.MAX_BETA})")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.VAE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.VAE_EPOCHS, eta_min=1e-6)

    save_path = os.path.join(config.OUTPUT_FOLDER, 'cvae.pth')
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
        'recon_quality': [], 'beta': []
    }
    best_val_recon = float('inf')

    for epoch in range(config.VAE_EPOCHS):
        # KL annealing: 前50%保持0，后50%线性增长到max_beta
        warmup = int(config.VAE_EPOCHS * 0.5)
        if epoch < warmup:
            current_beta = 0.0
        else:
            current_beta = config.MAX_BETA * (epoch - warmup) / (config.VAE_EPOCHS - warmup)

        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(batch_x, batch_y)
            loss, recon_loss, kl_loss = vae_loss(x_recon, batch_x, mu, logvar, beta=current_beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses['total'] += loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['kl'] += kl_loss.item()
            n_batches += 1

        train_losses = {k: v / n_batches for k, v in train_losses.items()}
        scheduler.step()

        # 验证
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        val_corr, n_val, n_corr = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
                x_recon, mu, logvar, _ = model(batch_x, batch_y)
                loss, recon_loss, kl_loss = vae_loss(x_recon, batch_x, mu, logvar, beta=current_beta)
                val_losses['total'] += loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()
                n_val += 1
                for i in range(batch_x.size(0)):
                    corr = np.corrcoef(
                        batch_x[i].cpu().numpy().flatten(),
                        x_recon[i].cpu().numpy().flatten()
                    )[0, 1]
                    if not np.isnan(corr):
                        val_corr += corr
                        n_corr += 1

        val_losses = {k: v / max(n_val, 1) for k, v in val_losses.items()}
        val_corr /= max(n_corr, 1)

        history['train_loss'].append(train_losses['total'])
        history['train_recon'].append(train_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_loss'].append(val_losses['total'])
        history['val_recon'].append(val_losses['recon'])
        history['val_kl'].append(val_losses['kl'])
        history['recon_quality'].append(val_corr)
        history['beta'].append(current_beta)

        if val_losses['recon'] < best_val_recon:
            best_val_recon = val_losses['recon']
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{config.VAE_EPOCHS}] \u03b2={current_beta:.6f} "
                  f"Recon: {val_losses['recon']:.1f} | KL: {val_losses['kl']:.1f} | Corr: {val_corr:.4f}")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"\n  阶段2完成! 最佳Val Recon: {best_val_recon:.1f}")
    return history


# ============================================================================
# 可视化
# ============================================================================

def plot_training_history(ae_history, vae_history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(ae_history['train_recon'], label='Train', alpha=0.8)
    axes[0, 0].plot(ae_history['val_recon'], label='Val', alpha=0.8)
    axes[0, 0].set_title('Stage 1 (AE): Reconstruction Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ae_history['val_corr'], color='green', alpha=0.8)
    axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Target > 0.7')
    axes[0, 1].set_title('Stage 1: Reconstruction Correlation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    best_corr = max(ae_history['val_corr']) if ae_history['val_corr'] else 0
    axes[0, 2].text(0.5, 0.5,
                     f"Stage 1 Summary\n\n"
                     f"Final Train Recon: {ae_history['train_recon'][-1]:.1f}\n"
                     f"Final Val Recon: {ae_history['val_recon'][-1]:.1f}\n"
                     f"Final Val Corr: {ae_history['val_corr'][-1]:.4f}\n"
                     f"Best Val Corr: {best_corr:.4f}",
                     transform=axes[0, 2].transAxes, fontsize=12,
                     verticalalignment='center', horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[0, 2].set_title('Stage 1: Summary')
    axes[0, 2].axis('off')

    axes[1, 0].plot(vae_history['train_recon'], label='Train', alpha=0.8)
    axes[1, 0].plot(vae_history['val_recon'], label='Val', alpha=0.8)
    axes[1, 0].set_title('Stage 2 (VAE): Reconstruction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(vae_history['train_kl'], label='Train KL', alpha=0.8)
    axes[1, 1].plot(vae_history['val_kl'], label='Val KL', alpha=0.8)
    ax_beta = axes[1, 1].twinx()
    ax_beta.plot(vae_history['beta'], 'r--', label='Beta', alpha=0.5)
    ax_beta.set_ylabel('Beta', color='r')
    axes[1, 1].set_title('Stage 2: KL Divergence & Beta')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(loc='upper left')
    ax_beta.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(vae_history['recon_quality'], color='green', alpha=0.8)
    axes[1, 2].axhline(y=0.7, color='r', linestyle='--', label='Target > 0.7')
    axes[1, 2].set_title('Stage 2: Reconstruction Correlation')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylim(-0.1, 1.1)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Two-Stage Training History (Conv1D-VAE v3)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reconstruction(model, X, y, device, save_path, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    indices = []
    for label in [0, 1]:
        label_idx = np.where(y == label)[0]
        n = min(num_samples // 2, len(label_idx))
        if n > 0:
            indices.extend(np.random.choice(label_idx, n, replace=False))
    indices = indices[:num_samples]

    for i, idx in enumerate(indices):
        sample_x = torch.FloatTensor(X[idx:idx + 1]).to(device)
        sample_y = torch.LongTensor(y[idx:idx + 1]).to(device)

        with torch.no_grad():
            recon_x, _, _, _ = model(sample_x, sample_y)
            recon_x = recon_x.cpu().numpy()[0]

        orig = X[idx]
        corr = np.corrcoef(orig.flatten(), recon_x.flatten())[0, 1]

        channels = [0, 1, 6, 11]
        names = ['Thumb Acc X', 'Thumb Acc Y', 'Index Acc X', 'Index Gyro Z']

        for j, (ch, name) in enumerate(zip(channels, names)):
            axes[i, j].plot(orig[:, ch], 'b-', label='Original', alpha=0.7, linewidth=1)
            axes[i, j].plot(recon_x[:, ch], 'r--', label='Reconstructed', alpha=0.7, linewidth=1)
            axes[i, j].set_title(f'{name} (UPDRS={y[idx]}, corr={corr:.3f})')
            if i == 0:
                axes[i, j].legend(fontsize=8)
            axes[i, j].grid(True, alpha=0.3)

    plt.suptitle('Conv1D-VAE Reconstruction (v3)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_latent_space(model, X, y, device, save_path):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X).to(device)
        labels = torch.LongTensor(y).to(device)
        z, _, _ = model.encode(x, labels)
        # z: (B, 64, 45) → 取全局平均池化降维用于可视化
        z_pooled = z.mean(dim=2).cpu().numpy()  # (B, 64)

    perplexity = min(30, len(X) - 1)
    print("  正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    z_2d = tsne.fit_transform(z_pooled)

    plt.figure(figsize=(10, 8))
    for label, name, color in [(0, 'UPDRS 0', 'blue'), (1, 'UPDRS 1', 'red')]:
        mask = y == label
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=name, alpha=0.6, s=50)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Latent Space Visualization (Global Avg Pool)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_real_vs_synthetic(real_samples, synthetic_samples, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time = np.arange(real_samples.shape[1]) / 60
    ch_names = ['Thumb Acc X', 'Thumb Acc Y', 'Thumb Acc Z',
                'Thumb Gyro X', 'Thumb Gyro Y', 'Thumb Gyro Z']
    for i in range(6):
        ax = axes[i // 3, i % 3]
        for j in range(min(3, len(real_samples))):
            ax.plot(time, real_samples[j, :, i], 'b-', alpha=0.3,
                    label='Real' if j == 0 else '')
        for j in range(min(3, len(synthetic_samples))):
            ax.plot(time, synthetic_samples[j, :, i], 'r--', alpha=0.3,
                    label='Synthetic' if j == 0 else '')
        ax.set_title(ch_names[i])
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('Real vs Synthetic Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    config = Config()

    print("=" * 70)
    print("        Conv1D-VAE 训练 (v3 - 全卷积 + 空间潜在空间)")
    print("=" * 70)
    print(f"\n核心改进:")
    print(f"  - 全卷积编解码器（替代LSTM），适合并行时序重构")
    print(f"  - 空间潜在变量: z=(B, {config.LATENT_CHANNELS}, 45)")
    print(f"  - 有效潜在维度: {config.LATENT_CHANNELS}×45 = {config.LATENT_CHANNELS * 45}")
    print(f"    （远大于信号的~504个有效自由度）")
    print(f"  - 渐进式3层下采样: 360→180→90→45")
    print(f"  - 两阶段: AE {config.AE_EPOCHS}ep → VAE {config.VAE_EPOCHS}ep")

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # 加载数据
    print("\n[1/5] 加载数据...")
    train_path = os.path.join(config.DATA_FOLDER, 'train_data.npz')
    if not os.path.exists(train_path):
        print(f"错误: 找不到 {train_path}，请先运行 pd_preprocessing.py")
        return

    train_data = np.load(train_path)
    val_data = np.load(os.path.join(config.DATA_FOLDER, 'val_data.npz'))

    X_train = train_data['X'].astype(np.float32)
    y_train = train_data['y']
    X_val = val_data['X'].astype(np.float32)
    y_val = val_data['y']

    print(f"  训练集: {X_train.shape}")
    print(f"    UPDRS 0: {(y_train == 0).sum()}, UPDRS 1: {(y_train == 1).sum()}")
    print(f"  验证集: {X_val.shape}")
    print(f"  数据范围: [{X_train.min():.2f}, {X_train.max():.2f}]")

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=min(config.BATCH_SIZE, len(val_dataset)), shuffle=False)

    # 创建模型
    print("\n[2/5] 创建模型...")
    model = ConvCVAE(
        input_dim=config.INPUT_DIM,
        latent_channels=config.LATENT_CHANNELS,
        num_classes=2,
        seq_len=config.SEQ_LEN
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")
    print(f"  压缩率: {360 * 12} → {config.LATENT_CHANNELS * 45} = {360 * 12 / (config.LATENT_CHANNELS * 45):.1f}:1")

    # 快速检查前向传播
    with torch.no_grad():
        test_x = torch.randn(2, 360, 12).to(config.DEVICE)
        test_y = torch.tensor([0, 1]).to(config.DEVICE)
        out, mu, lv, z = model(test_x, test_y)
        print(f"  前向检查: input={test_x.shape} → recon={out.shape}, z={z.shape}")

    # 训练
    print("\n[3/5] 训练模型...")
    ae_history = train_stage1_ae(model, train_loader, val_loader, config)
    vae_history = train_stage2_vae(model, train_loader, val_loader, config)

    # 训练曲线
    plot_training_history(ae_history, vae_history,
                          os.path.join(config.OUTPUT_FOLDER, 'training_history.png'))
    print(f"  训练曲线已保存")

    # 加载最佳VAE模型
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_FOLDER, 'cvae.pth'), weights_only=True))
    model.eval()

    # 可视化
    print("\n[4/5] 可视化...")
    plot_reconstruction(model, X_train, y_train, config.DEVICE,
                        os.path.join(config.OUTPUT_FOLDER, 'reconstruction.png'))
    print(f"  重构图已保存")

    plot_latent_space(model, X_train, y_train, config.DEVICE,
                      os.path.join(config.OUTPUT_FOLDER, 'latent_space.png'))
    print(f"  潜在空间图已保存")

    # 生成合成数据
    print("\n[5/5] 生成合成数据...")
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()

    if n0 != n1:
        minority = 0 if n0 < n1 else 1
        diff = abs(n0 - n1)
        print(f"  生成 {diff} 个 UPDRS {minority}分 的合成样本...")

        with torch.no_grad():
            labels = torch.tensor([minority] * diff).to(config.DEVICE)
            synthetic = model.generate(labels, device=config.DEVICE)
            synthetic = synthetic.cpu().numpy()

        real_minority = X_train[y_train == minority]
        plot_real_vs_synthetic(real_minority, synthetic,
                               os.path.join(config.OUTPUT_FOLDER, 'real_vs_synthetic.png'))
        print(f"  真实vs合成对比图已保存")

        X_aug = np.concatenate([X_train, synthetic], axis=0)
        y_aug = np.concatenate([y_train, np.full(diff, minority)], axis=0)
        idx = np.random.permutation(len(X_aug))
        X_aug, y_aug = X_aug[idx], y_aug[idx]

        np.savez_compressed(
            os.path.join(config.OUTPUT_FOLDER, 'augmented_train_data.npz'),
            X=X_aug, y=y_aug
        )
        print(f"  增强后: {X_aug.shape}, UPDRS 0: {(y_aug == 0).sum()}, 1: {(y_aug == 1).sum()}")
    else:
        print("  数据已平衡，跳过生成")

    print("\n" + "=" * 70)
    print("                      全部完成!")
    print("=" * 70)
    print(f"\n输出目录: {config.OUTPUT_FOLDER}")
    print("""
文件列表:
  ├── cvae_ae_pretrain.pth        # 阶段1预训练模型
  ├── cvae.pth                    # 最终VAE模型
  ├── training_history.png        # 训练曲线
  ├── reconstruction.png          # 重构效果图
  ├── latent_space.png            # 潜在空间t-SNE
  ├── real_vs_synthetic.png       # 真实vs合成对比
  └── augmented_train_data.npz    # 增强后的训练数据
""")


if __name__ == "__main__":
    main()