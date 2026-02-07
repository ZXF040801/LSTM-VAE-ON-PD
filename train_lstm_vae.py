"""
LSTM-VAE 训练脚本 (改进版 v2)
==========================
修复重构效果差的问题 - 单文件版本

主要改进:
1. 使用非自回归解码器，避免Teacher Forcing问题
2. 更温和的KL annealing
3. 添加重构质量监控（相关系数）
4. 更稳定的训练过程

运行: python train_lstm_vae_v2.py
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
    # 数据路径 - 请根据实际情况修改
    DATA_FOLDER = r'D:\Final work\DataForStudentProject-HWU\processed_data'
    OUTPUT_FOLDER = r'D:\Final work\DataForStudentProject-HWU\vae_results'

    # 模型参数
    INPUT_DIM = 12
    HIDDEN_DIM = 128
    LATENT_DIM = 32
    SEQ_LEN = 360
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 200
    LEARNING_RATE = 5e-4

    # VAE参数
    MAX_BETA = 0.1  # 大幅降低KL权重，更关注重构

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 改进的模型定义
# ============================================================================

class ImprovedEncoder(nn.Module):
    """改进的编码器"""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        h = F.relu(self.fc(hidden_cat))
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class ImprovedDecoder(nn.Module):
    """
    改进的解码器 - 非自回归方式
    将z复制到每个时间步，直接并行输出整个序列
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.2):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 将z扩展
        self.fc_expand = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, target_seq=None, teacher_forcing_ratio=0.0):
        batch_size = z.size(0)

        # 将z扩展到整个序列长度
        z_expanded = self.fc_expand(z)
        z_seq = z_expanded.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 添加位置编码
        positions = torch.linspace(-1, 1, self.seq_len, device=z.device)
        pos_enc = positions.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, self.hidden_dim)
        z_seq = z_seq + 0.1 * pos_enc

        # LSTM解码
        lstm_out, _ = self.lstm(z_seq)
        output = self.fc_output(lstm_out)

        return output


class ImprovedCVAE(nn.Module):
    """改进的条件LSTM-VAE"""

    def __init__(self, input_dim=12, hidden_dim=128, latent_dim=32, seq_len=360,
                 num_classes=2, num_layers=2, dropout=0.2):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, 16)
        self.encoder = ImprovedEncoder(input_dim + 16, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = ImprovedDecoder(latent_dim + 16, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels, teacher_forcing_ratio=0.0):
        class_emb = self.class_embedding(labels)
        class_emb_seq = class_emb.unsqueeze(1).repeat(1, x.size(1), 1)

        x_cond = torch.cat([x, class_emb_seq], dim=2)
        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)

        z_cond = torch.cat([z, class_emb], dim=1)
        x_recon = self.decoder(z_cond)

        return x_recon, mu, logvar, z

    def encode(self, x, labels=None):
        if labels is None:
            labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        class_emb = self.class_embedding(labels)
        class_emb_seq = class_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_cond = torch.cat([x, class_emb_seq], dim=2)

        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def generate(self, labels, num_samples=None, device='cpu'):
        if isinstance(labels, int):
            labels = torch.tensor([labels] * num_samples).to(device)

        batch_size = labels.size(0)
        z = torch.randn(batch_size, self.latent_dim).to(device)
        class_emb = self.class_embedding(labels)
        z_cond = torch.cat([z, class_emb], dim=1)

        return self.decoder(z_cond)


# ============================================================================
# 损失函数
# ============================================================================

def vae_loss_improved(x_recon, x, mu, logvar, beta=1.0):
    """改进的VAE损失"""
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ============================================================================
# Beta调度
# ============================================================================

def get_beta(epoch, total_epochs, max_beta=0.1):
    """温和的KL annealing"""
    warmup_start = int(total_epochs * 0.4)
    warmup_end = int(total_epochs * 0.8)

    if epoch < warmup_start:
        return 0.0
    elif epoch > warmup_end:
        return max_beta
    else:
        progress = (epoch - warmup_start) / (warmup_end - warmup_start)
        return max_beta * progress


# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader, config):
    """训练函数"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=25, factor=0.5, min_lr=1e-6
    )

    save_path = os.path.join(config.OUTPUT_FOLDER, 'cvae.pth')

    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
        'recon_quality': []
    }

    best_val_recon = float('inf')

    print(f"\n开始训练...")
    print(f"  设备: {config.DEVICE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Max Beta: {config.MAX_BETA}")
    print("-" * 70)

    for epoch in range(config.EPOCHS):
        current_beta = get_beta(epoch, config.EPOCHS, config.MAX_BETA)

        # 训练
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(batch_x, batch_y)
            loss, recon_loss, kl_loss = vae_loss_improved(x_recon, batch_x, mu, logvar, current_beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses['total'] += loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['kl'] += kl_loss.item()

        n_train = len(train_loader)
        train_losses = {k: v / n_train for k, v in train_losses.items()}

        # 验证
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}
        recon_quality = 0
        n_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)

                x_recon, mu, logvar, _ = model(batch_x, batch_y)
                loss, recon_loss, kl_loss = vae_loss_improved(x_recon, batch_x, mu, logvar, current_beta)

                val_losses['total'] += loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()

                # 计算重构质量
                for i in range(batch_x.size(0)):
                    orig = batch_x[i].cpu().numpy().flatten()
                    recon = x_recon[i].cpu().numpy().flatten()
                    corr = np.corrcoef(orig, recon)[0, 1]
                    if not np.isnan(corr):
                        recon_quality += corr
                        n_samples += 1

        n_val = len(val_loader)
        val_losses = {k: v / n_val for k, v in val_losses.items()}
        recon_quality = recon_quality / max(n_samples, 1)

        scheduler.step(val_losses['recon'])

        # 记录历史
        history['train_loss'].append(train_losses['total'])
        history['train_recon'].append(train_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_loss'].append(val_losses['total'])
        history['val_recon'].append(val_losses['recon'])
        history['val_kl'].append(val_losses['kl'])
        history['recon_quality'].append(recon_quality)

        # 保存最佳模型（基于重构损失）
        if val_losses['recon'] < best_val_recon:
            best_val_recon = val_losses['recon']
            torch.save(model.state_dict(), save_path)
            status = " ★"
        else:
            status = ""

        # 打印日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1:3d}/{config.EPOCHS}] β={current_beta:.4f} LR={current_lr:.6f}")
            print(f"  Train Recon: {train_losses['recon']:.4f} | Val Recon: {val_losses['recon']:.4f}")
            print(f"  Recon Corr: {recon_quality:.4f}{status}")

    print("-" * 70)
    print(f"训练完成! 最佳验证重构损失: {best_val_recon:.4f}")

    return history


# ============================================================================
# 可视化函数
# ============================================================================

def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_recon'], label='Train', alpha=0.8)
    axes[0, 1].plot(history['val_recon'], label='Val', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_kl'], label='Train', alpha=0.8)
    axes[1, 0].plot(history['val_kl'], label='Val', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['recon_quality'], label='Correlation', color='green', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Reconstruction Quality (Target > 0.7)')
    axes[1, 1].axhline(y=0.7, color='r', linestyle='--', label='Good threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reconstruction(model, X, y, device, save_path, num_samples=4):
    """可视化重构效果"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    indices = np.random.choice(len(X), num_samples, replace=False)

    for i, idx in enumerate(indices):
        sample_x = torch.FloatTensor(X[idx:idx + 1]).to(device)
        sample_y = torch.LongTensor(y[idx:idx + 1]).to(device)

        with torch.no_grad():
            recon_x, _, _, _ = model(sample_x, sample_y)
            recon_x = recon_x.cpu().numpy()[0]

        orig = X[idx]

        # 计算相关系数
        corr = np.corrcoef(orig.flatten(), recon_x.flatten())[0, 1]

        channels = [0, 1, 6, 11]
        names = ['Thumb Acc X', 'Thumb Acc Y', 'Index Acc X', 'Index Gyro Z']

        for j, (ch, name) in enumerate(zip(channels, names)):
            axes[i, j].plot(orig[:, ch], 'b-', label='Original', alpha=0.7, linewidth=1)
            axes[i, j].plot(recon_x[:, ch], 'r--', label='Reconstructed', alpha=0.7, linewidth=1)
            axes[i, j].set_title(f'{name} (UPDRS={y[idx]}, corr={corr:.2f})')
            if i == 0:
                axes[i, j].legend(fontsize=8)
            axes[i, j].grid(True, alpha=0.3)

    plt.suptitle('LSTM-VAE Reconstruction (Improved v2)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_latent_space(model, X, y, device, save_path):
    """可视化潜在空间"""
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(X).to(device)
        labels = torch.LongTensor(y).to(device)
        z, _, _ = model.encode(x, labels)
        z = z.cpu().numpy()

    print("  正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(10, 8))

    for label, name, color in [(0, 'UPDRS 0', 'blue'), (1, 'UPDRS 1', 'red')]:
        mask = y == label
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=name, alpha=0.6, s=50)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Latent Space Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_real_vs_synthetic(real_samples, synthetic_samples, save_path):
    """对比真实和合成样本"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time = np.arange(real_samples.shape[1]) / 60

    channel_names = ['Thumb Acc X', 'Thumb Acc Y', 'Thumb Acc Z',
                     'Thumb Gyro X', 'Thumb Gyro Y', 'Thumb Gyro Z']

    for i in range(6):
        ax = axes[i // 3, i % 3]

        for j in range(min(3, len(real_samples))):
            ax.plot(time, real_samples[j, :, i], 'b-', alpha=0.3,
                    label='Real' if j == 0 else '')

        for j in range(min(3, len(synthetic_samples))):
            ax.plot(time, synthetic_samples[j, :, i], 'r--', alpha=0.3,
                    label='Synthetic' if j == 0 else '')

        ax.set_title(channel_names[i])
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
    print("        LSTM-VAE 训练 (改进版 v2)")
    print("=" * 70)
    print(f"\n改进内容:")
    print(f"  - 非自回归解码器（避免Teacher Forcing问题）")
    print(f"  - 温和的KL annealing (max_beta={config.MAX_BETA})")
    print(f"  - 重构质量监控（相关系数）")
    print(f"  - 基于重构损失选择最佳模型")

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # 加载数据
    print("\n[1/5] 加载数据...")
    if not os.path.exists(os.path.join(config.DATA_FOLDER, 'train_data.npz')):
        print(f"错误: 找不到数据文件，请检查路径: {config.DATA_FOLDER}")
        return

    train_data = np.load(os.path.join(config.DATA_FOLDER, 'train_data.npz'))
    val_data = np.load(os.path.join(config.DATA_FOLDER, 'val_data.npz'))

    X_train, y_train = train_data['X'].astype(np.float32), train_data['y']
    X_val, y_val = val_data['X'].astype(np.float32), val_data['y']

    print(f"  训练集: {X_train.shape}")
    print(f"    UPDRS 0: {(y_train == 0).sum()}, UPDRS 1: {(y_train == 1).sum()}")
    print(f"  验证集: {X_val.shape}")

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=min(config.BATCH_SIZE, len(val_dataset)),
                            shuffle=False, drop_last=False)

    # 创建模型
    print("\n[2/5] 创建模型...")
    model = ImprovedCVAE(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        seq_len=config.SEQ_LEN,
        num_classes=2,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("\n[3/5] 训练模型...")
    history = train_model(model, train_loader, val_loader, config)

    # 绘制训练历史
    plot_training_history(history, os.path.join(config.OUTPUT_FOLDER, 'training_history.png'))
    print(f"  训练曲线已保存")

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_FOLDER, 'cvae.pth')))
    model.eval()

    # 可视化重构
    print("\n[4/5] 可视化...")
    plot_reconstruction(model, X_train, y_train, config.DEVICE,
                        os.path.join(config.OUTPUT_FOLDER, 'reconstruction.png'))
    print(f"  重构图已保存")

    # 潜在空间
    plot_latent_space(model, X_train, y_train, config.DEVICE,
                      os.path.join(config.OUTPUT_FOLDER, 'latent_space.png'))
    print(f"  潜在空间图已保存")

    # 生成合成数据
    print("\n[5/5] 生成合成数据...")
    num_class_0 = (y_train == 0).sum()
    num_class_1 = (y_train == 1).sum()

    if num_class_0 != num_class_1:
        minority_class = 0 if num_class_0 < num_class_1 else 1
        diff = abs(num_class_0 - num_class_1)

        print(f"  生成 {diff} 个 UPDRS {minority_class}分 的合成样本...")

        with torch.no_grad():
            labels = torch.tensor([minority_class] * diff).to(config.DEVICE)
            synthetic = model.generate(labels, device=config.DEVICE)
            synthetic = synthetic.cpu().numpy()

        # 可视化真实vs合成
        real_minority = X_train[y_train == minority_class]
        plot_real_vs_synthetic(real_minority, synthetic,
                               os.path.join(config.OUTPUT_FOLDER, 'real_vs_synthetic.png'))
        print(f"  真实vs合成对比图已保存")

        X_augmented = np.concatenate([X_train, synthetic], axis=0)
        y_augmented = np.concatenate([y_train, np.full(diff, minority_class)], axis=0)

        # 打乱
        indices = np.random.permutation(len(X_augmented))
        X_augmented, y_augmented = X_augmented[indices], y_augmented[indices]

        np.savez_compressed(
            os.path.join(config.OUTPUT_FOLDER, 'augmented_train_data.npz'),
            X=X_augmented, y=y_augmented
        )

        print(f"  增强后训练集: {X_augmented.shape}")
        print(f"    UPDRS 0: {(y_augmented == 0).sum()}, UPDRS 1: {(y_augmented == 1).sum()}")
    else:
        print("  数据已平衡，跳过生成")

    print("\n" + "=" * 70)
    print("                      全部完成!")
    print("=" * 70)
    print(f"\n输出目录: {config.OUTPUT_FOLDER}")
    print("""
文件列表:
  ├── cvae.pth                 # 训练好的模型
  ├── training_history.png        # 训练曲线（含重构质量）
  ├── reconstruction.png          # 重构效果图
  ├── latent_space.png           # 潜在空间t-SNE
  ├── real_vs_synthetic.png      # 真实vs合成对比
  └── augmented_train_data.npz   # 增强后的训练数据
""")


if __name__ == "__main__":
    main()