"""
LSTM-VAE 训练脚本 (UPDRS版)
===========================
用于UPDRS 0分 vs 1分的分类数据增强

运行: python train_lstm_vae_updrs.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys

# 导入LSTM-VAE模块
from lstm_vae import (
    LSTMVAE,
    ConditionalLSTMVAE,
    generate_synthetic_samples,
    plot_training_history,
    plot_latent_space,
    vae_loss
)


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    # ====== 数据路径 (UPDRS版) ======
    DATA_FOLDER = r'D:\Final work\DataForStudentProject-HWU\processed_data'
    OUTPUT_FOLDER = r'D:\Final work\DataForStudentProject-HWU\vae_results'
    # ================================

    # 模型参数
    INPUT_DIM = 12
    HIDDEN_DIM = 128
    LATENT_DIM = 32
    SEQ_LEN = 360
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    MAX_BETA = 1.0

    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# 调度器函数
# ============================================================================

def get_beta(epoch, total_epochs):
    """KL Annealing"""
    warmup_start = int(total_epochs * 0.2)
    warmup_end = int(total_epochs * 0.5)

    if epoch < warmup_start:
        return 0.0
    elif epoch > warmup_end:
        return 1.0
    else:
        return (epoch - warmup_start) / (warmup_end - warmup_start)


def get_tf_ratio(epoch, total_epochs):
    """Teacher Forcing Schedule"""
    decay_end = int(total_epochs * 0.6)

    if epoch >= decay_end:
        return 0.0
    else:
        return 1.0 - (epoch / decay_end)


# ============================================================================
# 训练函数
# ============================================================================

def train_vae_model(model, train_loader, val_loader, config, model_type='cvae'):
    """VAE训练函数"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6, verbose=True
    )

    save_path = os.path.join(config.OUTPUT_FOLDER, f'{model_type}.pth')

    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': []
    }

    best_val_loss = float('inf')

    print(f"\n开始训练 {model_type.upper()}...")
    print(f"  设备: {config.DEVICE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print("-" * 60)

    for epoch in range(config.EPOCHS):
        current_beta = get_beta(epoch, config.EPOCHS) * config.MAX_BETA
        current_tf = get_tf_ratio(epoch, config.EPOCHS)

        # 训练阶段
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}

        for batch_data in train_loader:
            if model_type == 'cvae':
                x, y = batch_data
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            else:
                x, _ = batch_data
                x = x.to(config.DEVICE)
                y = None

            optimizer.zero_grad()

            if model_type == 'cvae':
                x_recon, mu, logvar, _ = model(x, y, teacher_forcing_ratio=current_tf)
            else:
                x_recon, mu, logvar, _ = model(x, teacher_forcing_ratio=current_tf)

            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=current_beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses['total'] += loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['kl'] += kl_loss.item()

        n_train = len(train_loader)
        train_losses = {k: v / n_train for k, v in train_losses.items()}

        # 验证阶段
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}

        with torch.no_grad():
            for batch_data in val_loader:
                if model_type == 'cvae':
                    x, y = batch_data
                    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                else:
                    x, _ = batch_data
                    x = x.to(config.DEVICE)
                    y = None

                if model_type == 'cvae':
                    x_recon, mu, logvar, _ = model(x, y, teacher_forcing_ratio=0.0)
                else:
                    x_recon, mu, logvar, _ = model(x, teacher_forcing_ratio=0.0)

                loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=current_beta)

                val_losses['total'] += loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()

        n_val = len(val_loader)
        val_losses = {k: v / n_val for k, v in val_losses.items()}

        scheduler.step(val_losses['total'])

        # 记录历史
        history['train_loss'].append(train_losses['total'])
        history['train_recon'].append(train_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_loss'].append(val_losses['total'])
        history['val_recon'].append(val_losses['recon'])
        history['val_kl'].append(val_losses['kl'])

        # 保存最佳模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(model.state_dict(), save_path)
            status = " ★ saved"
        else:
            status = ""

        # 打印日志
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1:3d}/{config.EPOCHS}] "
                  f"β={current_beta:.2f} TF={current_tf:.2f} LR={current_lr:.6f}")
            print(f"  Train - Loss: {train_losses['total']:.3f} | "
                  f"Recon: {train_losses['recon']:.3f} | KL: {train_losses['kl']:.4f}")
            print(f"  Val   - Loss: {val_losses['total']:.3f} | "
                  f"Recon: {val_losses['recon']:.3f} | KL: {val_losses['kl']:.4f}{status}")

    print("-" * 60)
    print(f"{model_type.upper()} 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    return history


# ============================================================================
# 主流程
# ============================================================================

def main():
    config = Config()

    print("=" * 60)
    print("      LSTM-VAE 训练 (UPDRS 0分 vs 1分)")
    print("=" * 60)
    print(f"\n分类任务:")
    print(f"  - label=0: UPDRS 0分 (正常)")
    print(f"  - label=1: UPDRS 1分 (轻度异常)")
    print(f"\n设备: {config.DEVICE}")

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # ========== 1. 加载数据 ==========
    print("\n[1/5] 加载数据...")

    train_path = os.path.join(config.DATA_FOLDER, 'train_data.npz')
    if not os.path.exists(train_path):
        print(f"错误: 找不到数据文件")
        print(f"请先运行 pd_preprocessing_updrs.py")
        print(f"检查路径: {config.DATA_FOLDER}")
        return

    train_data = np.load(train_path)
    val_data = np.load(os.path.join(config.DATA_FOLDER, 'val_data.npz'))
    test_data = np.load(os.path.join(config.DATA_FOLDER, 'test_data.npz'))

    X_train, y_train = train_data['X'].astype(np.float32), train_data['y']
    X_val, y_val = val_data['X'].astype(np.float32), val_data['y']
    X_test, y_test = test_data['X'].astype(np.float32), test_data['y']

    print(f"  训练集: {X_train.shape}")
    print(f"    UPDRS 0分: {(y_train == 0).sum()}")
    print(f"    UPDRS 1分: {(y_train == 1).sum()}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    # ========== 2. 创建 DataLoader ==========
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # ========== 3. 训练条件 LSTM-VAE ==========
    print("\n[2/5] 训练 Conditional LSTM-VAE...")

    cvae_model = ConditionalLSTMVAE(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        seq_len=config.SEQ_LEN,
        num_classes=2,  # 0分和1分
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    print(f"  模型参数量: {sum(p.numel() for p in cvae_model.parameters()):,}")

    cvae_history = train_vae_model(
        cvae_model, train_loader, val_loader, config, model_type='cvae'
    )

    # 绘制训练历史
    plot_training_history(
        cvae_history,
        save_path=os.path.join(config.OUTPUT_FOLDER, 'cvae_training_history.png')
    )

    # ========== 4. 生成合成数据 ==========
    print("\n[3/5] 生成合成数据...")

    cvae_model.load_state_dict(
        torch.load(os.path.join(config.OUTPUT_FOLDER, 'cvae.pth'))
    )
    cvae_model.eval()

    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)

    print(f"  当前训练集: UPDRS 0分={num_class_0}, UPDRS 1分={num_class_1}")

    if num_class_0 < num_class_1:
        minority_class = 0
        diff = num_class_1 - num_class_0
        minority_name = "UPDRS 0分"
    else:
        minority_class = 1
        diff = num_class_0 - num_class_1
        minority_name = "UPDRS 1分"

    if diff > 0:
        print(f"  生成 {diff} 个 {minority_name} 的合成样本...")

        synthetic_samples = generate_synthetic_samples(
            cvae_model, diff, class_label=minority_class, device=config.DEVICE
        )

        X_augmented = np.concatenate([X_train, synthetic_samples], axis=0)
        y_augmented = np.concatenate([y_train, np.full(diff, minority_class)], axis=0)

        # 打乱
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]

        np.savez_compressed(
            os.path.join(config.OUTPUT_FOLDER, 'augmented_train_data.npz'),
            X=X_augmented, y=y_augmented
        )

        print(f"  增强后训练集: {X_augmented.shape}")
        print(f"    UPDRS 0分: {(y_augmented == 0).sum()}")
        print(f"    UPDRS 1分: {(y_augmented == 1).sum()}")
    else:
        print("  数据集已平衡，跳过生成。")

    # ========== 5. 可视化 ==========
    print("\n[4/5] 可视化...")

    # 重构对比
    sample_idx = 0
    sample_x = torch.FloatTensor(X_train[sample_idx:sample_idx + 1]).to(config.DEVICE)
    sample_y = torch.LongTensor(y_train[sample_idx:sample_idx + 1]).to(config.DEVICE)

    cvae_model.eval()
    with torch.no_grad():
        recon_x, _, _, _ = cvae_model(sample_x, sample_y, teacher_forcing_ratio=0.0)
        recon_x_np = recon_x.cpu().numpy().squeeze(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    channel_names = ['Thumb Acc X', 'Thumb Acc Y', 'Index Acc X', 'Index Gyro Z']
    channel_indices = [0, 1, 6, 11]

    for i, (ax, ch_name, ch_idx) in enumerate(zip(axes.ravel(), channel_names, channel_indices)):
        ax.plot(X_train[sample_idx, :, ch_idx], label='Original', alpha=0.8)
        ax.plot(recon_x_np[:, ch_idx], '--', label='Reconstructed', alpha=0.8)
        ax.set_title(f"{ch_name} (UPDRS={y_train[sample_idx]})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time steps')

    plt.suptitle('LSTM-VAE Reconstruction (UPDRS Classification)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_FOLDER, 'reconstruction_check.png'), dpi=150)
    plt.close()

    # t-SNE
    print("\n[5/5] 可视化潜在空间 (t-SNE)...")
    if len(X_train) < 3000:
        plot_latent_space(
            cvae_model, X_train, y_train, device=config.DEVICE,
            save_path=os.path.join(config.OUTPUT_FOLDER, 'latent_space.png')
        )
    else:
        indices = np.random.choice(len(X_train), 2000, replace=False)
        plot_latent_space(
            cvae_model, X_train[indices], y_train[indices], device=config.DEVICE,
            save_path=os.path.join(config.OUTPUT_FOLDER, 'latent_space.png')
        )

    print("\n" + "=" * 60)
    print("                   全部完成!")
    print("=" * 60)
    print(f"\n输出文件保存在: {config.OUTPUT_FOLDER}")
    print("""
  ├── cvae.pth                    # 训练好的模型
  ├── augmented_train_data.npz    # 增强后的训练数据
  ├── cvae_training_history.png   # 训练曲线
  ├── reconstruction_check.png    # 重构对比图
  └── latent_space.png            # 潜在空间可视化
""")


if __name__ == "__main__":
    main()