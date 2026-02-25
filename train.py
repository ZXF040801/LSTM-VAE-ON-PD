"""
训练模块：

  Phase 1 - train_cvae()         : 训练 Conditional LSTM-VAE
  Phase 2 - generate_synthetic() : 用训练好的 CVAE 生成合成少数类序列
  Phase 3 - train_classifier()   : 训练 LSTM 二分类器（支持真实/增强数据）
  工具函数 - eval_clf_metrics()   : 单次评估，返回 loss/acc/f1
"""

import os
import numpy as np
import torch
import torch.nn as nn

from model import vae_loss


# ─── Phase 1: 训练 CVAE ──────────────────────────────────────────────────────

def train_cvae(model, train_loader, val_loader, device,
               epochs=100, lr=1e-3, kl_anneal_epochs=20,
               save_path='cvae_best.pth'):
    """
    训练 Conditional LSTM-VAE。

    关键技巧：
      - KL 退火 (beta annealing)：前 kl_anneal_epochs 个 epoch beta 从 0 线性增至 1
        防止训练初期 KL 项主导，导致 posterior collapse
      - Teacher Forcing 训练，推理时关闭
      - 梯度裁剪 (clip_grad_norm, max=1.0)
      - ReduceLROnPlateau 学习率调度
      - 按验证集损失保存最优模型

    Returns: history dict (train_loss, val_loss, recon_loss, kl_loss)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5, verbose=False)

    best_val   = float('inf')
    history    = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}

    print(f"  开始训练 CVAE，共 {epochs} epochs，KL 退火期: {kl_anneal_epochs}")

    for epoch in range(1, epochs + 1):

        # KL 退火系数：线性从 0 增到 1
        beta = min(1.0, epoch / max(kl_anneal_epochs, 1))

        # ── 训练 ──────────────────────────────────────────────────────────
        model.train()
        tr_total, tr_recon, tr_kl = 0.0, 0.0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(x, y, teacher_forcing=True)
            loss, r_l, kl_l  = vae_loss(recon, x, mu, logvar, beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_total += loss.item()
            tr_recon += r_l.item()
            tr_kl    += kl_l.item()

        n_tr = len(train_loader)
        tr_total /= n_tr
        tr_recon /= n_tr
        tr_kl    /= n_tr

        # ── 验证 ──────────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                recon, mu, logvar = model(x, y, teacher_forcing=False)
                loss, _, _        = vae_loss(recon, x, mu, logvar, beta=1.0)
                val_total        += loss.item()
        val_total /= len(val_loader)

        scheduler.step(val_total)

        history['train_loss'].append(tr_total)
        history['val_loss'].append(val_total)
        history['recon_loss'].append(tr_recon)
        history['kl_loss'].append(tr_kl)

        # 保存最优
        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1:
            lr_cur = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}/{epochs}  β={beta:.2f}  lr={lr_cur:.2e}"
                  f"  Train={tr_total:.4f}(R={tr_recon:.4f},KL={tr_kl:.4f})"
                  f"  Val={val_total:.4f}")

    print(f"  CVAE 训练完成，最佳验证损失: {best_val:.4f} → {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return history


# ─── Phase 2: 生成合成序列 ───────────────────────────────────────────────────

def generate_synthetic(model, label, n_samples, device):
    """
    用训练好的 CVAE 生成 n_samples 条指定类别的合成序列。
    返回归一化空间中的 numpy 数组 (n_samples, SEQ_LEN, C)。

    注意：合成序列已在归一化空间，可直接与归一化后的真实序列拼接。
    """
    synth = model.generate(label, device, n_samples)   # Tensor
    return synth.cpu().numpy().astype(np.float32)


# ─── Phase 3: 训练分类器 ─────────────────────────────────────────────────────

def train_classifier(model, train_loader, val_loader, device,
                     pos_weight, epochs=80, lr=1e-3,
                     save_path='clf_best.pth'):
    """
    训练 LSTM 二分类器。

    处理类别不平衡：BCEWithLogitsLoss 的 pos_weight = n0/n1
    学习率调度：CosineAnnealingLR（平滑衰减，避免震荡）
    按验证集 Macro-F1 保存最优模型（比 Accuracy 更适合不平衡数据）

    Returns: history dict (train_loss, val_loss, val_acc, val_f1)
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    best_f1  = 0.0
    history  = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print(f"  开始训练分类器，共 {epochs} epochs，pos_weight={pos_weight.item():.2f}")

    for epoch in range(1, epochs + 1):

        # ── 训练 ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # ── 验证 ──────────────────────────────────────────────────────────
        val_loss, val_acc, val_f1 = eval_clf_metrics(
            model, val_loader, device, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs}"
                  f"  Train Loss={tr_loss:.4f}"
                  f"  Val Loss={val_loss:.4f}"
                  f"  Acc={val_acc:.3f}"
                  f"  F1={val_f1:.3f}")

    print(f"  分类器训练完成，最佳验证 F1: {best_f1:.4f} → {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return history


# ─── 工具：单次评估 ──────────────────────────────────────────────────────────

def eval_clf_metrics(model, loader, device, criterion=None, threshold=0.5):
    """
    计算一次完整 pass 的 loss / accuracy / macro-F1。

    Returns: (loss, accuracy, f1_macro)
    """
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_preds, all_probs, all_true = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y.float().to(device)).item()
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob >= threshold).astype(int)
            all_probs.extend(prob.tolist())
            all_preds.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)
    loss = total_loss / max(len(loader), 1)

    return loss, acc, f1
