"""
主运行脚本 — Parkinson's Disease UPDRS 0 vs 1 分类

流程：
  1. 数据加载与预处理（从 Excel + data/ 文件夹）
  2. 训练 Conditional LSTM-VAE
  3. 生成合成 UPDRS=1 少数类序列
  4a. 训练基线 LSTM 分类器（仅真实数据）
  4b. 训练增强 LSTM 分类器（真实 + 合成数据）
  5. 评估并保存可视化结果

运行：
  python main.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import prepare_data, PDDataset, SEQ_LEN, INPUT_DIM
from model import LSTM_CVAE, LSTMClassifier
from train import train_cvae, train_classifier, generate_synthetic
from evaluate import (
    full_evaluate,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
    tsne_visualization,
    plot_signal_comparison,
)

# ─── 配置 ─────────────────────────────────────────────────────────────────────

BASE_DIR = '/home/user/LSTM-VAE-ON-PD'
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED     = 42

# 模型超参数
HIDDEN_DIM    = 64
LATENT_DIM    = 16
NUM_LAYERS    = 2
LABEL_EMB_DIM = 4
DROPOUT       = 0.2

# 训练超参数
BATCH_SIZE       = 32
CVAE_EPOCHS      = 100
CLF_EPOCHS       = 80
LR_CVAE          = 1e-3
LR_CLF           = 1e-3
KL_ANNEAL_EPOCHS = 25   # KL 退火轮数

# 数据增强：生成多少条合成少数类样本
N_SYNTH = 60            # 生成 60 条 UPDRS=1，使训练集接近平衡


# ─── 工具 ────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def out(filename):
    return os.path.join(OUT_DIR, filename)


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  PD UPDRS 0 vs 1 — LSTM-CVAE 数据增强实验")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: 数据准备
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 1] 数据加载与预处理")
    data = prepare_data(
        BASE_DIR,
        test_size=0.2,
        val_size=0.1,
        random_state=SEED,
        batch_size=BATCH_SIZE,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: 训练 Conditional LSTM-VAE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 2] 训练 Conditional LSTM-VAE")
    cvae = LSTM_CVAE(
        input_dim     = INPUT_DIM,
        hidden_dim    = HIDDEN_DIM,
        latent_dim    = LATENT_DIM,
        num_layers    = NUM_LAYERS,
        seq_len       = SEQ_LEN,
        num_classes   = 2,
        label_emb_dim = LABEL_EMB_DIM,
        dropout       = DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in cvae.parameters() if p.requires_grad)
    print(f"  CVAE 可训练参数: {n_params:,}")

    hist_cvae = train_cvae(
        cvae,
        data['train_loader'],
        data['val_loader'],
        DEVICE,
        epochs           = CVAE_EPOCHS,
        lr               = LR_CVAE,
        kl_anneal_epochs = KL_ANNEAL_EPOCHS,
        save_path        = out('cvae_best.pth'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: 生成合成 UPDRS=1 序列
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[Phase 3] 生成 {N_SYNTH} 条合成 UPDRS=1 序列")
    synth_X = generate_synthetic(cvae, label=1, n_samples=N_SYNTH, device=DEVICE)
    synth_y = np.ones(N_SYNTH, dtype=np.int64)
    print(f"  合成序列形状: {synth_X.shape}")

    # 可视化合成质量（均值曲线对比）
    plot_signal_comparison(
        data['X_train'], data['y_train'], synth_X,
        channel_names=['dist', 's1_pitch', 's1_roll', 's1_x', 's1_z', 's2_y'],
        save_path=out('signal_comparison.png'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4a: 基线分类器（仅真实数据）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 4a] 训练基线分类器（仅真实数据）")
    clf_base = LSTMClassifier(
        input_dim  = INPUT_DIM,
        hidden_dim = HIDDEN_DIM,
        num_layers = NUM_LAYERS,
        dropout    = 0.3,
    ).to(DEVICE)

    hist_base = train_classifier(
        clf_base,
        data['train_loader'],
        data['val_loader'],
        DEVICE,
        data['pos_weight'],
        epochs    = CLF_EPOCHS,
        lr        = LR_CLF,
        save_path = out('clf_base.pth'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4b: 增强分类器（真实 + 合成数据）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 4b] 训练增强分类器（真实 + 合成数据）")

    aug_dataset = ConcatDataset([
        PDDataset(data['X_train'], data['y_train']),
        PDDataset(synth_X, synth_y),
    ])
    aug_loader = DataLoader(
        aug_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 更新 pos_weight（增强后训练集的类别权重）
    n0_aug = int(np.sum(data['y_train'] == 0))
    n1_aug = int(np.sum(data['y_train'] == 1)) + N_SYNTH
    pos_weight_aug = torch.tensor([n0_aug / max(n1_aug, 1)], dtype=torch.float32)
    print(f"  增强后训练集: n0={n0_aug}, n1={n1_aug}, pos_weight={pos_weight_aug.item():.3f}")

    clf_aug = LSTMClassifier(
        input_dim  = INPUT_DIM,
        hidden_dim = HIDDEN_DIM,
        num_layers = NUM_LAYERS,
        dropout    = 0.3,
    ).to(DEVICE)

    hist_aug = train_classifier(
        clf_aug,
        aug_loader,
        data['val_loader'],
        DEVICE,
        pos_weight_aug,
        epochs    = CLF_EPOCHS,
        lr        = LR_CLF,
        save_path = out('clf_aug.pth'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5: 评估
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 5] 测试集评估")

    trues_b, preds_b, probs_b, auc_b = full_evaluate(
        clf_base, data['test_loader'], DEVICE, '基线分类器 (Real only)')

    trues_a, preds_a, probs_a, auc_a = full_evaluate(
        clf_aug, data['test_loader'], DEVICE, '增强分类器 (Real + Synth)')

    # 混淆矩阵
    plot_confusion_matrix(trues_b, preds_b,
                          title='Baseline Classifier',
                          save_path=out('cm_baseline.png'))
    plot_confusion_matrix(trues_a, preds_a,
                          title='Augmented Classifier',
                          save_path=out('cm_augmented.png'))

    # ROC 曲线
    plot_roc_curves([
        (trues_b, probs_b, 'Baseline (Real only)'),
        (trues_a, probs_a, 'Augmented (Real+Synth)'),
    ], save_path=out('roc_comparison.png'))

    # 训练历史
    plot_training_curves(
        hist_cvae, hist_base, hist_aug,
        save_path=out('training_history.png'),
    )

    # t-SNE 可视化（用训练集的一部分 + 合成样本）
    print("\n[Phase 5] t-SNE 隐空间可视化")
    rng  = np.random.default_rng(SEED)
    n_vis = min(120, len(data['X_train']))
    idx  = rng.choice(len(data['X_train']), n_vis, replace=False)

    tsne_visualization(
        cvae,
        data['X_train'][idx], data['y_train'][idx],
        synth_X[:60], synth_y[:60],
        DEVICE,
        save_path=out('tsne.png'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  实验结果汇总")
    print('='*60)
    print(f"  {'模型':<30} {'ROC-AUC':>10}")
    print(f"  {'-'*42}")
    print(f"  {'基线分类器 (Real only)':<30} {auc_b:>10.4f}")
    print(f"  {'增强分类器 (Real+Synth)':<30} {auc_a:>10.4f}")
    print(f"  {'AUC 变化量':<30} {(auc_a - auc_b):>+10.4f}")
    print('='*60)
    print(f"\n  所有输出文件保存在: {OUT_DIR}/")
    files = [
        'cvae_best.pth       — CVAE 最优权重',
        'clf_base.pth        — 基线分类器权重',
        'clf_aug.pth         — 增强分类器权重',
        'signal_comparison   — 真实 vs 合成信号均值对比',
        'cm_baseline.png     — 基线混淆矩阵',
        'cm_augmented.png    — 增强混淆矩阵',
        'roc_comparison.png  — ROC 曲线对比',
        'training_history    — 训练曲线',
        'tsne.png            — t-SNE 隐空间可视化',
    ]
    for f in files:
        print(f"    {f}")
    print()


if __name__ == '__main__':
    main()
