"""
评估与可视化模块：

  full_evaluate()         : 完整测试集评估（分类报告 + ROC-AUC）
  plot_confusion_matrix() : 混淆矩阵
  plot_roc_curves()       : 多模型 ROC 曲线对比
  plot_training_curves()  : CVAE 损失 + 分类器训练曲线
  tsne_visualization()    : t-SNE 可视化真实 vs 合成样本隐空间分布
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 无 GUI 环境下使用非交互后端
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)
from sklearn.manifold import TSNE


# ─── 完整评估 ────────────────────────────────────────────────────────────────

def full_evaluate(model, loader, device, name=''):
    """
    对 loader 中的所有数据进行完整评估。

    打印：classification_report + ROC-AUC
    Returns: (y_true, y_pred, y_prob, auc)
    """
    model.eval()
    preds, probs, trues = [], [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            prob   = torch.sigmoid(logits).cpu().numpy()
            pred   = (prob >= 0.5).astype(int)
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            trues.extend(y.numpy().tolist())

    trues = np.array(trues)
    preds = np.array(preds)
    probs = np.array(probs)

    auc = roc_auc_score(trues, probs)

    title = f"评估结果：{name}" if name else "评估结果"
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print('=' * 55)
    print(classification_report(
        trues, preds,
        target_names=['UPDRS=0 (正常)', 'UPDRS=1 (轻度PD)'],
        digits=4,
    ))
    print(f"  ROC-AUC : {auc:.4f}")
    print('=' * 55)

    return trues, preds, probs, auc


# ─── 混淆矩阵 ────────────────────────────────────────────────────────────────

def plot_confusion_matrix(trues, preds, title='', save_path=None):
    """绘制并保存混淆矩阵。"""
    cm   = confusion_matrix(trues, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['UPDRS=0', 'UPDRS=1'])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  混淆矩阵已保存: {save_path}")
    plt.close(fig)


# ─── ROC 曲线 ────────────────────────────────────────────────────────────────

def plot_roc_curves(results, save_path=None):
    """
    多模型 ROC 曲线对比。

    results: list of (y_true, y_prob, label_str)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    for i, (trues, probs, label) in enumerate(results):
        fpr, tpr, _ = roc_curve(trues, probs)
        auc = roc_auc_score(trues, probs)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2, label=f"{label}  AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title('ROC Curve Comparison', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ROC 曲线已保存: {save_path}")
    plt.close(fig)


# ─── 训练曲线 ─────────────────────────────────────────────────────────────────

def plot_training_curves(hist_cvae, hist_base, hist_aug, save_path=None):
    """
    绘制三块图：
      左: CVAE 训练/验证损失（含 recon 和 KL 分量）
      中: 分类器验证 F1（基线 vs 增强）
      右: 分类器验证 Accuracy（基线 vs 增强）
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- CVAE Loss ---
    ax = axes[0]
    ax.plot(hist_cvae['train_loss'], label='Train Total',  color='#2196F3')
    ax.plot(hist_cvae['val_loss'],   label='Val Total',    color='#2196F3', linestyle='--')
    ax.plot(hist_cvae['recon_loss'], label='Train Recon',  color='#4CAF50', alpha=0.7)
    ax.plot(hist_cvae['kl_loss'],    label='Train KL',     color='#FF9800', alpha=0.7)
    ax.set_title('CVAE Training Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Classifier F1 ---
    ax = axes[1]
    ax.plot(hist_base['val_f1'], label='Baseline (Real only)', color='#F44336')
    ax.plot(hist_aug['val_f1'],  label='Augmented (Real+Synth)', color='#2196F3')
    ax.set_title('Classifier Validation F1 (macro)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # --- Classifier Accuracy ---
    ax = axes[2]
    ax.plot(hist_base['val_acc'], label='Baseline', color='#F44336')
    ax.plot(hist_aug['val_acc'],  label='Augmented', color='#2196F3')
    ax.set_title('Classifier Validation Accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle('Training History', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  训练曲线已保存: {save_path}")
    plt.close(fig)


# ─── t-SNE 可视化 ────────────────────────────────────────────────────────────

def tsne_visualization(cvae_model, X_real, y_real, X_synth, y_synth,
                       device, save_path=None):
    """
    将真实样本和合成样本的 CVAE 隐向量投影到 2D，用 t-SNE 可视化。

    颜色编码：
      蓝色圆点  - Real UPDRS=0
      红色方块  - Real UPDRS=1
      橙色三角  - Synthetic UPDRS=1

    返回: (z_2d, all_labels)
    """
    cvae_model.eval()

    def get_mu(X, y):
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.LongTensor(y).to(device)
        return cvae_model.encode(X_t, y_t).cpu().numpy()

    mu_real  = get_mu(X_real,  y_real)
    mu_synth = get_mu(X_synth, y_synth)

    all_z      = np.vstack([mu_real, mu_synth])
    all_labels = (
        ['Real-0'] * int(np.sum(y_real == 0)) +
        ['Real-1'] * int(np.sum(y_real == 1)) +
        ['Synth-1'] * len(y_synth)
    )
    all_labels = np.array(all_labels)

    print(f"  t-SNE 降维中（样本数={len(all_z)}）...")
    perp  = min(30, max(5, len(all_z) // 5))
    tsne  = TSNE(n_components=2, perplexity=perp,
                 random_state=42, n_iter=1000, verbose=0)
    z_2d  = tsne.fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(8, 6))
    style = {
        'Real-0':  ('o', '#2196F3', 0.7),
        'Real-1':  ('s', '#F44336', 0.8),
        'Synth-1': ('^', '#FF9800', 0.7),
    }
    for lbl, (marker, color, alpha) in style.items():
        mask = all_labels == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=color, marker=marker, alpha=alpha,
                   s=55, edgecolors='none', label=lbl)

    ax.set_title('t-SNE: Latent Space — Real vs Synthetic', fontsize=12)
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  t-SNE 图已保存: {save_path}")
    plt.close(fig)

    return z_2d, all_labels


# ─── 合成质量检验：序列统计对比 ──────────────────────────────────────────────

def plot_signal_comparison(X_real, y_real, X_synth, n_channels=6,
                           channel_names=None, save_path=None):
    """
    对每个通道绘制真实 UPDRS=1 样本与合成样本的均值±标准差曲线，
    用于直观检验合成质量。
    """
    if channel_names is None:
        channel_names = ['dist', 's1_pitch', 's1_roll', 's1_x', 's1_z', 's2_y']

    real_1 = X_real[y_real == 1]
    synth  = X_synth

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()

    for i in range(min(n_channels, len(channel_names))):
        ax = axes[i]
        t  = np.arange(real_1.shape[1])

        # 真实
        m_r = real_1[:, :, i].mean(axis=0)
        s_r = real_1[:, :, i].std(axis=0)
        ax.plot(t, m_r, color='#F44336', linewidth=1.5, label='Real UPDRS=1')
        ax.fill_between(t, m_r - s_r, m_r + s_r, color='#F44336', alpha=0.2)

        # 合成
        m_s = synth[:, :, i].mean(axis=0)
        s_s = synth[:, :, i].std(axis=0)
        ax.plot(t, m_s, color='#FF9800', linewidth=1.5,
                linestyle='--', label='Synthetic UPDRS=1')
        ax.fill_between(t, m_s - s_s, m_s + s_s, color='#FF9800', alpha=0.2)

        ax.set_title(channel_names[i], fontsize=10)
        ax.set_xlabel('Time step'); ax.set_ylabel('Normalized value')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Signal Comparison: Real vs Synthetic (mean ± std)', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  信号对比图已保存: {save_path}")
    plt.close(fig)
