import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================================
# 1. LSTM-VAE 模型定义
# ============================================================================

class LSTMEncoder(nn.Module):
    """
    LSTM编码器: 将输入序列编码到潜在空间

    输入: (batch_size, seq_len, input_dim)
    输出: mu, logvar (batch_size, latent_dim)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.2):
        super(LSTMEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 双向LSTM
        )

        # 从LSTM隐藏状态到潜在空间
        # 双向LSTM输出维度是 hidden_dim * 2
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)

        # LSTM编码
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers * 2, batch_size, hidden_dim) for bidirectional

        # 取最后一层的前向和后向隐藏状态并拼接
        hidden_forward = hidden[-2, :, :]  # (batch_size, hidden_dim)
        hidden_backward = hidden[-1, :, :]  # (batch_size, hidden_dim)
        hidden_cat = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch_size, hidden_dim*2)

        # 计算潜在分布的参数
        mu = self.fc_mu(hidden_cat)
        logvar = self.fc_logvar(hidden_cat)

        return mu, logvar


class LSTMDecoder(nn.Module):
    """
    修改后的解码器：支持 Teacher Forcing
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=2, dropout=0.2):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 将潜在向量 z 映射到隐藏状态
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # 输入层：现在输入是 (当前真实值/预测值 + 潜在向量 z)
        # 我们把 z 拼接到每个时间步的输入上，增强 z 的影响力
        self.lstm = nn.LSTM(
            input_size=output_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, target_seq=None, teacher_forcing_ratio=0.5):
        """
        参数:
            z: 潜在向量 (batch, latent_dim)
            target_seq: 真实序列 (batch, seq_len, input_dim)，训练时提供
        """
        batch_size = z.size(0)

        # 1. 初始化隐藏状态
        hidden = self.fc_hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        cell = self.fc_cell(z).view(self.num_layers, batch_size, self.hidden_dim)

        # 2. 准备输入
        outputs = []

        # 第一个时间步的输入是全0 (GO token)
        # device = z.device
        input_step = torch.zeros(batch_size, 1, self.output_dim).to(z.device)

        # 将 z 扩展以便在每一步拼接
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)

        for t in range(self.seq_len):
            # 拼接 input + z
            lstm_input = torch.cat([input_step, z_expanded], dim=2)

            # LSTM 单步前向
            out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

            # 预测输出
            pred = self.fc_output(out)  # (batch, 1, output_dim)
            outputs.append(pred)

            # 决定下一步的输入 (Teacher Forcing)
            if target_seq is not None and np.random.random() < teacher_forcing_ratio:
                # 训练模式：有一定概率使用真实值作为下一步输入
                input_step = target_seq[:, t:t + 1, :]
            else:
                # 推理模式：使用当前预测作为下一步输入
                input_step = pred

        return torch.cat(outputs, dim=1)


class LSTMVAE(nn.Module):
    """
    完整的LSTM-VAE模型
    """

    def __init__(self, input_dim=12, hidden_dim=128, latent_dim=32, seq_len=360, num_layers=2, dropout=0.2):
        super(LSTMVAE, self).__init__()

        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # 编码器和解码器
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        x_recon = self.decoder(z)

        return x_recon, mu, logvar, z

    def encode(self, x):
        """只进行编码，返回潜在向量"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        """从潜在向量解码"""
        return self.decoder(z)

    def generate(self, num_samples, device='cpu'):
        """从先验分布生成新样本"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


# ============================================================================
# 2. 条件LSTM-VAE (Conditional LSTM-VAE)
# ============================================================================

class ConditionalLSTMVAE(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, latent_dim=32, seq_len=360,
                 num_classes=2, num_layers=2, dropout=0.2):
        super(ConditionalLSTMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, 8)  # 稍微减小嵌入维度

        # 编码器 (输入维度 = 原始维度 + 类别嵌入维度)
        # 我们把类别拼接到输入数据上
        self.encoder = LSTMEncoder(input_dim + 8, hidden_dim, latent_dim, num_layers, dropout)

        # 解码器
        self.decoder = LSTMDecoder(latent_dim + 8, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels, teacher_forcing_ratio=0.5):
        # 1. 处理标签
        class_emb = self.class_embedding(labels)  # (batch, 8)

        # 将标签扩展到序列长度并拼接到输入 x
        class_emb_seq = class_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_cond = torch.cat([x, class_emb_seq], dim=2)

        # 2. 编码
        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)

        # 3. 拼接标签到 z
        z_cond = torch.cat([z, class_emb], dim=1)

        # 4. 解码 (传入原始 x 用于 Teacher Forcing)
        x_recon = self.decoder(z_cond, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)

        return x_recon, mu, logvar, z

    def generate(self, labels, num_samples=None, device='cpu'):
        if isinstance(labels, int):
            labels = torch.tensor([labels] * num_samples).to(device)

        batch_size = labels.size(0)
        z = torch.randn(batch_size, self.latent_dim).to(device)
        class_emb = self.class_embedding(labels)
        z_cond = torch.cat([z, class_emb], dim=1)

        # 生成时不传入 target_seq，teacher_forcing_ratio 自动无效
        return self.decoder(z_cond, target_seq=None)

# ============================================================================
# 3. 损失函数
# ============================================================================

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE损失函数修改版
    使用 sum reduction 防止损失值过小导致梯度消失
    """
    batch_size = x.size(0)

    # 1. 重构损失 (MSE) - 使用 sum 而不是 mean
    # 这样损失值与序列长度和特征维度成正比，梯度更强
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # 2. KL散度
    # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 归一化处理：为了方便观察，我们除以 batch_size
    # 这样 Loss 不会因为 batch_size 变化而剧烈变化
    recon_loss = recon_loss / batch_size
    kl_loss = kl_loss / batch_size

    # 总损失
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


# ============================================================================
# 4. 训练函数
# ============================================================================

# 在 train_lstm_vae.py 中

def get_beta(epoch, total_epochs, cycle_epochs=None):
    """
    计算 KL 退火的 beta 值 (Cyclical Annealing 或 线性增长)
    这里使用简单的线性增长策略：前 20% 的 epoch beta=0，
    中间 50% 线性增长到 1，最后 30% 保持为 1。
    """
    if epoch < total_epochs * 0.2:
        return 0.0
    elif epoch > total_epochs * 0.7:
        return 1.0
    else:
        # 线性增长区间
        start = total_epochs * 0.2
        end = total_epochs * 0.7
        return (epoch - start) / (end - start)


def train_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, beta=1.0,
              device='cuda', save_path='models/lstm_vae.pth'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 稍微降低 patience，如果不动就早点降学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ===== 计算当前的 Beta (KL 退火) =====
        # 注意：如果你想用固定的 beta，就注释掉下面这行，用参数传入的 beta
        current_beta = get_beta(epoch, epochs)

        # ===== 训练阶段 =====
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_recon, mu, logvar, _ = model(batch_x)

            # 使用动态 beta
            loss, recon_loss, kl_loss = vae_loss(x_recon, batch_x, mu, logvar, beta=current_beta)

            loss.backward()
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses['total'] += loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['kl'] += kl_loss.item()

        n_batches = len(train_loader)
        train_losses = {k: v / n_batches for k, v in train_losses.items()}

        # ===== 验证阶段 =====
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}

        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                x_recon, mu, logvar, _ = model(batch_x)
                # 验证时也用同样的 beta 以便比较 loss
                loss, recon_loss, kl_loss = vae_loss(x_recon, batch_x, mu, logvar, beta=current_beta)

                val_losses['total'] += loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()

        n_val_batches = len(val_loader)
        val_losses = {k: v / n_val_batches for k, v in val_losses.items()}

        # 学习率调度
        scheduler.step(val_losses['total'])
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(train_losses['total'])
        history['train_recon'].append(train_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_loss'].append(val_losses['total'])
        history['val_recon'].append(val_losses['recon'])
        history['val_kl'].append(val_losses['kl'])

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Beta: {current_beta:.2f} LR: {current_lr:.6f}")
            print(
                f"  Train - Total: {train_losses['total']:.2f}, Recon: {train_losses['recon']:.2f}, KL: {train_losses['kl']:.4f}")
            print(
                f"  Val   - Total: {val_losses['total']:.2f}, Recon: {val_losses['recon']:.2f}, KL: {val_losses['kl']:.4f}")

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
    return history


def train_conditional_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, beta=1.0,
                          device='cuda', save_path='models/conditional_lstm_vae.pth'):
    """
    训练条件LSTM-VAE模型
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(batch_x, batch_y)
            loss, _, _ = vae_loss(x_recon, batch_x, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x_recon, mu, logvar, _ = model(batch_x, batch_y)
                loss, _, _ = vae_loss(x_recon, batch_x, mu, logvar, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")
    return history


# ============================================================================
# 5. 合成数据生成
# ============================================================================

def generate_synthetic_samples(model, num_samples, class_label=None, device='cpu'):
    """
    使用训练好的VAE生成合成样本

    参数:
        model: 训练好的VAE模型
        num_samples: 生成样本数量
        class_label: 类别标签 (仅用于条件VAE)
        device: 设备

    返回:
        生成的样本 (numpy array)
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        if isinstance(model, ConditionalLSTMVAE) and class_label is not None:
            # 条件生成
            labels = torch.tensor([class_label] * num_samples).to(device)
            samples = model.generate(labels, device=device)
        else:
            # 无条件生成
            samples = model.generate(num_samples, device=device)

    return samples.cpu().numpy()


def augment_minority_class(model, X_train, y_train, target_ratio=1.0, device='cpu'):
    """
    使用VAE生成合成样本来平衡数据集

    参数:
        model: 训练好的条件VAE
        X_train: 训练数据
        y_train: 训练标签
        target_ratio: 目标类别比例 (minority/majority)
        device: 设备

    返回:
        增强后的 X_train, y_train
    """
    # 统计类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]

    majority_count = counts.max()
    minority_count = counts.min()

    # 计算需要生成的样本数
    target_minority_count = int(majority_count * target_ratio)
    num_to_generate = target_minority_count - minority_count

    if num_to_generate <= 0:
        print("数据已平衡，无需生成")
        return X_train, y_train

    print(f"需要为类别 {minority_class} 生成 {num_to_generate} 个合成样本")

    # 生成合成样本
    synthetic_samples = generate_synthetic_samples(
        model, num_to_generate, class_label=minority_class, device=device
    )

    # 合并数据
    X_augmented = np.concatenate([X_train, synthetic_samples], axis=0)
    y_augmented = np.concatenate([y_train, np.full(num_to_generate, minority_class)], axis=0)

    # 打乱数据
    indices = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]

    print(f"增强后数据集大小: {len(X_augmented)}")
    print(f"类别分布: {dict(zip(*np.unique(y_augmented, return_counts=True)))}")

    return X_augmented, y_augmented


# ============================================================================
# 6. 可视化函数
# ============================================================================

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 总损失
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 重构损失
    if 'train_recon' in history:
        axes[1].plot(history['train_recon'], label='Train')
        axes[1].plot(history['val_recon'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # KL损失
    if 'train_kl' in history:
        axes[2].plot(history['train_kl'], label='Train')
        axes[2].plot(history['val_kl'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstruction(model, X_sample, device='cpu', save_path=None):
    """可视化重构效果"""
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        x = torch.FloatTensor(X_sample).unsqueeze(0).to(device)
        if isinstance(model, ConditionalLSTMVAE):
            # 对于条件VAE，使用标签0
            label = torch.tensor([0]).to(device)
            x_recon, _, _, _ = model(x, label)
        else:
            x_recon, _, _, _ = model(x)
        x_recon = x_recon.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time = np.arange(X_sample.shape[0]) / 60

    channel_names = ['thumb_acc_x', 'thumb_acc_y', 'thumb_acc_z',
                     'thumb_gyro_x', 'thumb_gyro_y', 'thumb_gyro_z']

    for i in range(6):
        ax = axes[i // 3, i % 3]
        ax.plot(time, X_sample[:, i], 'b-', label='Original', alpha=0.7)
        ax.plot(time, x_recon[:, i], 'r--', label='Reconstructed', alpha=0.7)
        ax.set_title(channel_names[i])
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Original vs Reconstructed Sequences', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_latent_space(model, X, y, device='cpu', save_path=None):
    """可视化潜在空间 (使用t-SNE降维)"""
    from sklearn.manifold import TSNE

    model.eval()
    model = model.to(device)

    # 获取潜在向量
    with torch.no_grad():
        x = torch.FloatTensor(X).to(device)
        z, _, _ = model.encode(x)
        z = z.cpu().numpy()

    # t-SNE降维
    print("正在进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z)

    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Class (0=Non-PD, 1=PD)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Latent Space Visualization (t-SNE)')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    print()

