import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_FOLDER = r"D:\Final work\DataForStudentProject-HWU\processed_data"
AUG_FOLDER  = r"D:\Final work\DataForStudentProject-HWU\vae_results"  # 里面有 augmented_train_data.npz

def seq_features(X):
    """
    X: (N, T, C)
    return: (N, F) 简单统计特征
    """
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    mx   = X.max(axis=1)
    mn   = X.min(axis=1)

    # 一阶差分能量（捕捉动作快慢/抖动）
    diff = np.diff(X, axis=1)
    diff_energy = (diff ** 2).mean(axis=1)

    # 原信号能量
    energy = (X ** 2).mean(axis=1)

    return np.concatenate([mean, std, mx, mn, diff_energy, energy], axis=1)

def train_and_report(train_npz, test_npz, title):
    tr = np.load(train_npz)
    te = np.load(test_npz)

    Xtr, ytr = tr["X"], tr["y"]
    Xte, yte = te["X"], te["y"]

    Ftr = seq_features(Xtr)
    Fte = seq_features(Xte)

    scaler = StandardScaler()
    Ftr = scaler.fit_transform(Ftr)
    Fte = scaler.transform(Fte)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Ftr, ytr)

    ypred = clf.predict(Fte)

    print("\n" + "="*60)
    print(title)
    print("="*60)
    print("Confusion matrix:\n", confusion_matrix(yte, ypred))
    print(classification_report(yte, ypred, target_names=["Non-PD", "PD"]))

if __name__ == "__main__":
    train_real = os.path.join(DATA_FOLDER, "train_data.npz")
    test_real  = os.path.join(DATA_FOLDER, "test_data.npz")

    # 1) Baseline: 真实数据训练
    train_and_report(train_real, test_real, "Baseline (Real train → Real test)")

    # 2) Augmented: 增强训练集训练
    train_aug = os.path.join(AUG_FOLDER, "augmented_train_data.npz")
    if os.path.exists(train_aug):
        train_and_report(train_aug, test_real, "Augmented (Real+Synthetic train → Real test)")
    else:
        print("\n[WARN] augmented_train_data.npz not found, skip augmented evaluation.")
