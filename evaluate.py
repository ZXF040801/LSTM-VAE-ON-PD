"""
VAE 训练效果评估脚本
==================
评估方式:
1. 基线: 直接用原始数据训练分类器
2. VAE潜在特征: 用编码器提取的潜在特征训练分类器
3. 数据增强: 用VAE生成的合成数据增强训练集后训练分类器

输出: precision / recall / f1-score / confusion matrix

运行: python evaluate_vae.py
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

class Config:
    DATA_FOLDER = r'D:\Final work\DataForStudentProject-HWU\processed_data'
    OUTPUT_FOLDER = r'D:\Final work\DataForStudentProject-HWU\vae_results'
    DEVICE = 'cpu'  # 评估用CPU即可


# ============================================================================
# 模型定义（与train_lstm_vae.py一致，评估时需要加载模型）
# ============================================================================

import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, input_dim=12, latent_channels=64, class_embed_dim=16):
        super().__init__()
        in_ch = input_dim + class_embed_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
        )
        self.conv_mu = nn.Conv1d(256, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv1d(256, latent_channels, kernel_size=1)

    def forward(self, x_cond):
        h = self.encoder(x_cond)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, output_dim=12, latent_channels=64, class_embed_dim=16):
        super().__init__()
        in_ch = latent_channels + class_embed_dim
        self.decoder = nn.Sequential(
            nn.Conv1d(in_ch, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, output_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z_cond):
        return self.decoder(z_cond)


class ConvCVAE(nn.Module):
    def __init__(self, input_dim=12, latent_channels=64, num_classes=2, seq_len=360):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_classes = num_classes
        self.class_embed_dim = 16
        self.spatial_len = seq_len // 8
        self.class_embedding = nn.Embedding(num_classes, self.class_embed_dim)
        self.encoder = ConvEncoder(input_dim, latent_channels, self.class_embed_dim)
        self.decoder = ConvDecoder(input_dim, latent_channels, self.class_embed_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x, labels):
        x_t = x.transpose(1, 2)
        class_emb = self.class_embedding(labels)
        class_spatial_in = class_emb.unsqueeze(2).expand(-1, -1, x_t.size(2))
        x_cond = torch.cat([x_t, class_spatial_in], dim=1)
        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)
        class_spatial_z = class_emb.unsqueeze(2).expand(-1, -1, z.size(2))
        z_cond = torch.cat([z, class_spatial_z], dim=1)
        x_recon_t = self.decoder(z_cond)
        return x_recon_t.transpose(1, 2), mu, logvar, z

    def encode(self, x, labels=None):
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
        if isinstance(labels, int):
            labels = torch.tensor([labels] * num_samples).to(device)
        z = torch.randn(labels.size(0), self.latent_channels, self.spatial_len).to(device)
        class_emb = self.class_embedding(labels)
        class_spatial_z = class_emb.unsqueeze(2).expand(-1, -1, z.size(2))
        z_cond = torch.cat([z, class_spatial_z], dim=1)
        return self.decoder(z_cond).transpose(1, 2)


# ============================================================================
# 特征提取函数
# ============================================================================

def extract_handcrafted_features(X):
    """从原始时序数据提取手工统计特征"""
    features = []
    for sample in X:
        f = []
        for ch in range(sample.shape[1]):
            s = sample[:, ch]
            f.extend([
                s.mean(), s.std(), s.min(), s.max(),
                np.percentile(s, 25), np.percentile(s, 75),
                np.sqrt(np.mean(s**2)),                      # RMS
                np.mean(np.abs(np.diff(s))),                 # 平均变化率
                np.sum(np.diff(np.sign(s)) != 0),            # 过零率
            ])
        features.append(f)
    return np.array(features)


def extract_latent_features(model, X, y, device='cpu'):
    """用VAE编码器提取潜在特征"""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X).to(device)
        labels = torch.LongTensor(y).to(device)
        z, mu, _ = model.encode(x, labels)
        # mu: (B, 64, 45) → 多种池化方式组合
        mu_mean = mu.mean(dim=2)   # 全局平均池化 (B, 64)
        mu_max = mu.max(dim=2)[0]  # 全局最大池化 (B, 64)
        mu_std = mu.std(dim=2)     # 全局标准差池化 (B, 64)
        features = torch.cat([mu_mean, mu_max, mu_std], dim=1)  # (B, 192)
    return features.cpu().numpy()


# ============================================================================
# 主评估流程
# ============================================================================

def main():
    config = Config()

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("=" * 70)
    print("           VAE 训练效果评估")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    print("\n[1] 加载数据...")
    train_data = np.load(os.path.join(config.DATA_FOLDER, 'train_data.npz'))
    val_data = np.load(os.path.join(config.DATA_FOLDER, 'val_data.npz'))
    test_data = np.load(os.path.join(config.DATA_FOLDER, 'test_data.npz'))

    X_train, y_train = train_data['X'].astype(np.float32), train_data['y']
    X_val, y_val = val_data['X'].astype(np.float32), val_data['y']
    X_test, y_test = test_data['X'].astype(np.float32), test_data['y']

    # 合并train+val作为完整训练集
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    print(f"  训练集: {X_train.shape} (UPDRS0: {(y_train==0).sum()}, UPDRS1: {(y_train==1).sum()})")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape} (UPDRS0: {(y_test==0).sum()}, UPDRS1: {(y_test==1).sum()})")
    print(f"  合并训练: {X_train_full.shape}")

    # 加载增强数据
    aug_path = os.path.join(config.OUTPUT_FOLDER, 'augmented_train_data.npz')
    if os.path.exists(aug_path):
        aug_data = np.load(aug_path)
        X_aug, y_aug = aug_data['X'].astype(np.float32), aug_data['y']
        print(f"  增强训练集: {X_aug.shape} (UPDRS0: {(y_aug==0).sum()}, UPDRS1: {(y_aug==1).sum()})")
    else:
        X_aug, y_aug = None, None
        print("  增强数据不存在，跳过增强评估")

    # ------------------------------------------------------------------
    # 2. 加载VAE模型
    # ------------------------------------------------------------------
    print("\n[2] 加载VAE模型...")
    model = ConvCVAE(input_dim=12, latent_channels=64, num_classes=2, seq_len=360)
    model_path = os.path.join(config.OUTPUT_FOLDER, 'cvae.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        print(f"  模型加载成功: {model_path}")
        vae_loaded = True
    else:
        print(f"  模型文件不存在: {model_path}")
        vae_loaded = False

    # ------------------------------------------------------------------
    # 3. 提取特征
    # ------------------------------------------------------------------
    print("\n[3] 提取特征...")

    # 方法A: 手工统计特征（基线）
    print("  提取手工统计特征...")
    feat_train_raw = extract_handcrafted_features(X_train_full)
    feat_test_raw = extract_handcrafted_features(X_test)
    print(f"    手工特征维度: {feat_train_raw.shape[1]}")

    # 方法B: 展平原始数据（另一个基线）
    feat_train_flat = X_train_full.reshape(len(X_train_full), -1)
    feat_test_flat = X_test.reshape(len(X_test), -1)
    print(f"    展平特征维度: {feat_train_flat.shape[1]}")

    # 方法C: VAE潜在特征
    if vae_loaded:
        print("  提取VAE潜在特征...")
        feat_train_vae = extract_latent_features(model, X_train_full, y_train_full)
        feat_test_vae = extract_latent_features(model, X_test, y_test)
        print(f"    VAE特征维度: {feat_train_vae.shape[1]}")

    # 方法D: 增强数据 + 手工特征
    if X_aug is not None:
        # 增强数据只增强训练集，用同样的手工特征
        # 把val也加到增强集里
        X_aug_full = np.concatenate([X_aug, X_val], axis=0)
        y_aug_full = np.concatenate([y_aug, y_val], axis=0)
        feat_train_aug = extract_handcrafted_features(X_aug_full)
        print(f"    增强后手工特征: {feat_train_aug.shape}")

    # ------------------------------------------------------------------
    # 4. 训练分类器并评估
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("           分类结果")
    print("=" * 70)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    target_names = ['Non-PD (0)', 'PD (1)']

    # 收集所有结果用于对比图
    all_results = {}

    def evaluate_approach(name, feat_train, y_tr, feat_test, y_te):
        print(f"\n{'─' * 60}")
        print(f"  方法: {name}")
        print(f"  训练样本: {len(feat_train)}, 测试样本: {len(feat_test)}")
        print(f"  特征维度: {feat_train.shape[1]}")
        print(f"{'─' * 60}")

        scaler = StandardScaler()
        feat_train_s = scaler.fit_transform(feat_train)
        feat_test_s = scaler.transform(feat_test)

        best_acc = 0
        best_clf_name = ''
        best_report = ''
        best_cm = None

        for clf_name, clf in classifiers.items():
            clf.fit(feat_train_s, y_tr)
            y_pred = clf.predict(feat_test_s)
            acc = accuracy_score(y_te, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_clf_name = clf_name
                best_report = classification_report(y_te, y_pred, target_names=target_names)
                best_cm = confusion_matrix(y_te, y_pred)

            # 简短输出每个分类器
            report_dict = classification_report(y_te, y_pred, target_names=target_names, output_dict=True)
            f1_0 = report_dict['Non-PD (0)']['f1-score']
            f1_1 = report_dict['PD (1)']['f1-score']
            print(f"    {clf_name:25s}  Acc={acc:.2f}  F1(0)={f1_0:.2f}  F1(1)={f1_1:.2f}")

        print(f"\n  ★ 最佳分类器: {best_clf_name} (Acc={best_acc:.2f})")
        print(best_report)

        all_results[name] = {
            'best_acc': best_acc,
            'best_clf': best_clf_name,
            'cm': best_cm,
        }

        return best_acc, best_cm

    # 评估各方法
    evaluate_approach(
        "A. 基线 - 手工统计特征",
        feat_train_raw, y_train_full, feat_test_raw, y_test
    )

    evaluate_approach(
        "B. 基线 - 展平原始数据",
        feat_train_flat, y_train_full, feat_test_flat, y_test
    )

    if vae_loaded:
        evaluate_approach(
            "C. VAE潜在特征",
            feat_train_vae, y_train_full, feat_test_vae, y_test
        )

    if X_aug is not None:
        evaluate_approach(
            "D. 增强数据 + 手工特征",
            feat_train_aug, y_aug_full, feat_test_raw, y_test
        )

        # 方法E: 增强数据 + VAE特征
        if vae_loaded:
            feat_aug_vae = extract_latent_features(model, X_aug_full, y_aug_full)
            evaluate_approach(
                "E. 增强数据 + VAE特征",
                feat_aug_vae, y_aug_full, feat_test_vae, y_test
            )

    # ------------------------------------------------------------------
    # 5. 可视化对比
    # ------------------------------------------------------------------
    print("\n[5] 生成对比图...")

    fig, axes = plt.subplots(1, len(all_results) + 1, figsize=(5 * (len(all_results) + 1), 4.5))

    # 混淆矩阵
    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        cm = res['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-PD', 'PD'], yticklabels=['Non-PD', 'PD'])
        short_name = name.split(' - ')[-1] if ' - ' in name else name
        ax.set_title(f'{short_name}\nAcc={res["best_acc"]:.2f}', fontsize=10)
        ax.set_ylabel('True' if idx == 0 else '')
        ax.set_xlabel('Predicted')

    # 准确率柱状图
    ax = axes[-1]
    names = [k.split('.')[1].strip() if '.' in k else k for k in all_results.keys()]
    accs = [v['best_acc'] for v in all_results.values()]
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(range(len(accs)), accs, color=colors[:len(accs)])
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('方法对比')
    ax.set_ylim(0, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('PD Classification: VAE Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_FOLDER, 'classification_evaluation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图已保存: {save_path}")

    # ------------------------------------------------------------------
    # 6. 总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                       总结")
    print("=" * 70)
    for name, res in all_results.items():
        print(f"  {name:40s} → Acc={res['best_acc']:.2f} ({res['best_clf']})")

    best_method = max(all_results.items(), key=lambda x: x[1]['best_acc'])
    print(f"\n  ★ 最佳方法: {best_method[0]}")
    print(f"    准确率: {best_method[1]['best_acc']:.2f}")
    print(f"    分类器: {best_method[1]['best_clf']}")


if __name__ == "__main__":
    main()