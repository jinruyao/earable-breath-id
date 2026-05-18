# main.py
"""
BreathSign 完整系统 (Full Pipeline)

系统流程:
    原始音频 → 降噪 → 呼吸事件检测 → 特征提取 → 身份认证

模块组成:
    Module 1: Noise Reduction (denoise.py)
    Module 2: Event Detection (event_detection.py)
    Module 3: Feature Extraction (feature_extraction.py)
    Module 4: Authentication (authentication.py)

使用方式:
    1. 训练模式: python main.py --mode train --data_dir <path>
    2. 测试模式: python main.py --mode test --data_dir <path>
    3. 演示模式: python main.py --mode demo --audio <path>
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 导入各模块
from denoise import BreathSignDenoiser
from event_detection import BreathEventDetector, visualize_detection
from feature_extraction import FeatureExtractor, extract_all_features
from authentication import (
    BreathSignAuthenticator,
    cross_validate,
    SVMAuthenticator,
)

# Matplotlib 中文字体配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC',
                                   'STHeiti', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================================================================
# 直接从 JSON 加载特征
# ================================================================
def load_features_from_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    features_by_user = {}
    for user_id, feats in raw.items():
        arr = np.array(feats, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        features_by_user[user_id] = arr

    print(f"[加载] 共 {len(features_by_user)} 个用户")
    for uid, arr in features_by_user.items():
        print(f"  {uid}: {arr.shape[0]} 样本, 维度 {arr.shape[1]}")

    return features_by_user


# ================================================================
# BreathSignSystem（不动）
# ================================================================
class BreathSignSystem:
    """
    BreathSign 完整系统
    整合所有模块，提供端到端的呼吸声身份认证功能。
    """

    def __init__(self):
        self.denoiser = BreathSignDenoiser()
        self.detector = None
        self.extractor = None
        self.authenticator = BreathSignAuthenticator()

    def load_audio(self, filepath):
        return self.denoiser.load_audio(filepath)

    def process_audio(self, filepath, apply_denoise=True):
        sr, left, right = self.load_audio(filepath)

        if self.detector is None or self.detector.sr != sr:
            self.detector = BreathEventDetector(sr=sr)
        if self.extractor is None or self.extractor.sr != sr:
            self.extractor = FeatureExtractor(sr=sr)

        if apply_denoise:
            left_denoised, _ = self.denoiser.denoise(left, sr)
            right_denoised = None
            if right is not None:
                right_denoised, _ = self.denoiser.denoise(right, sr)
        else:
            left_denoised = left
            right_denoised = right

        detection_result = self.detector.detect(left_denoised, sr)

        if len(detection_result['breath_cycles']) > 0:
            all_features, mean_features = self.extractor.extract_from_all_cycles(
                detection_result['breath_cycles'], left_denoised, right_denoised)
        else:
            mean_features, _ = self.extractor.extract_features(
                left_denoised, right_denoised, sr)
            all_features = mean_features.reshape(1, -1)

        return {
            'sr': sr,
            'is_stereo': right is not None,
            'duration': len(left) / sr,
            'n_breath_events': detection_result['n_events'],
            'n_breath_cycles': detection_result['n_cycles'],
            'breath_events': detection_result['breath_events'],
            'breath_cycles': detection_result['breath_cycles'],
            'all_features': all_features,
            'mean_features': mean_features,
            'final_features': mean_features,
            'left_denoised': left_denoised,
            'right_denoised': right_denoised,
            'detection_result': detection_result
        }

    def enroll_user(self, user_id, audio_files):
        all_features = []
        for filepath in audio_files:
            try:
                result = self.process_audio(filepath)
                if len(result['final_features']) > 0:
                    all_features.append(result['final_features'])
            except Exception as e:
                print(f"  Warning: Failed to process {filepath}: {e}")

        if len(all_features) == 0:
            return {'success': False, 'error': 'No valid features extracted'}

        all_features = np.array(all_features)
        template = self.authenticator.enroll(user_id, all_features)
        return {
            'success': True,
            'user_id': user_id,
            'n_samples': len(all_features),
            'feature_dim': len(template)
        }

    def authenticate_user(self, audio_file, claimed_user_id=None):
        process_result = self.process_audio(audio_file)
        auth_result = self.authenticator.authenticate(
            process_result['final_features'], claimed_user_id)
        return {
            **auth_result,
            'n_breath_cycles': process_result['n_breath_cycles'],
            'duration': process_result['duration']
        }

    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.authenticator.save(str(model_dir / 'authenticator.json'))
        config = {'use_triplet': False}
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def load(self, model_dir):
        model_dir = Path(model_dir)
        auth_path = model_dir / 'authenticator.json'
        if auth_path.exists():
            self.authenticator.load(str(auth_path))


# ================================================================
# compare_methods —— 核心新增函数
# ================================================================
def compare_methods(
    features_by_user: dict,
    train_features: dict,
    test_data: list,
    output_dir: Path,
    eer_euclidean: float,
    metrics_euclidean: dict,
    cv_mean_euclidean: dict,
):
    """
    在已有欧氏距离结果的基础上，训练 SVM 并做双方法对比。

    Parameters
    ----------
    features_by_user : dict
        全量特征 {user_id: ndarray(n, d)}，用于 SVM 交叉验证
    train_features : dict
        训练子集特征，用于 SVM 训练
    test_data : list[(feature, user_id)]
        测试集，用于 SVM 测试评估
    output_dir : Path
        结果保存目录
    eer_euclidean : float
        欧氏距离方法的 EER（已算好，直接传入）
    metrics_euclidean : dict
        欧氏距离方法的 evaluate() 返回值
    cv_mean_euclidean : dict
        欧氏距离方法的 cross_validate() 返回的 mean_metrics

    Returns
    -------
    comparison : dict
        两种方法的所有对比指标
    """

    # ────────────────────────────────────────────
    # Step 1: 训练 SVM
    # ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SVM 方法训练与评估")
    print("=" * 60)

    print("\n[SVM] 训练中 (RBF kernel, C=10, gamma=scale) ...")
    svm_auth = SVMAuthenticator(kernel='rbf', C=10.0, gamma='scale',
                                probability=True)
    svm_auth.train(train_features)
    print("[SVM] 训练完成")

    # ────────────────────────────────────────────
    # Step 2: SVM 测试集评估
    # ────────────────────────────────────────────
    print("\n[SVM] 测试集评估 ...")
    svm_metrics = svm_auth.evaluate(test_data, return_report=True)
    print(f"  测试准确率: {svm_metrics['accuracy']*100:.2f}%")
    print(f"  正确/总数:  {svm_metrics['n_correct']}/{svm_metrics['n_samples']}")
    if 'report' in svm_metrics:
        print("\n  分类报告:")
        for line in svm_metrics['report'].splitlines():
            print(f"    {line}")

    # ────────────────────────────────────────────
    # Step 3: SVM 交叉验证
    # ────────────────────────────────────────────
    print("\n[SVM] 5-折交叉验证 ...")
    svm_cv_mean, svm_cv_std, oof_true, oof_pred = svm_auth.cross_validate(
        features_by_user, n_folds=5, return_oof=True
    )
    print(f"  平均准确率: {svm_cv_mean*100:.2f}% ± {svm_cv_std*100:.2f}%")

    # ────────────────────────────────────────────
    # Step 4: 保存 SVM 模型
    # ────────────────────────────────────────────
    svm_model_dir = output_dir / 'model'
    svm_model_dir.mkdir(parents=True, exist_ok=True)
    svm_model_path = str(svm_model_dir / 'svm_model.pkl')
    svm_auth.save(svm_model_path)
    print(f"\n[SVM] 模型已保存: {svm_model_path}")

    # ────────────────────────────────────────────
    # Step 5: 汇总对比数据
    # ────────────────────────────────────────────
    comparison = {
        'euclidean': {
            'accuracy':    metrics_euclidean.get('accuracy', 0.0),
            'fnr':         metrics_euclidean.get('fnr', 0.0),
            'fpr':         metrics_euclidean.get('fpr', 0.0),
            'f1':          metrics_euclidean.get('f1', 0.0),
            'eer':         eer_euclidean,
            'cv_accuracy': cv_mean_euclidean.get('accuracy', 0.0),
            'cv_std':      cv_mean_euclidean.get('accuracy_std', 0.0),
        },
        'svm': {
            'accuracy':    svm_metrics['accuracy'],
            'fnr':         0.0,          # SVM 闭集识别不计 FNR/FPR，填 N/A
            'fpr':         0.0,
            'f1':          0.0,          # 整体 F1 需另算，此处留 0
            'eer':         float('nan'), # 闭集 SVM 不输出 EER
            'cv_accuracy': svm_cv_mean,
            'cv_std':      svm_cv_std,
        },
    }

    # 若需要 SVM 的 macro-F1，从 report 里解析（可选）
    if 'report' in svm_metrics:
        for line in svm_metrics['report'].splitlines():
            if 'macro avg' in line:
                parts = line.split()
                try:
                    # format: macro avg  prec  rec  f1  support
                    comparison['svm']['f1'] = float(parts[4])
                except Exception:
                    pass
    # ────────────────────────────────────────────
    # Step 5.5: 生成 SVM 混淆矩阵
    # ────────────────────────────────────────────
    print("\n[SVM] 生成混淆矩阵 ...")

    # 收集 test_data 的真实标签和预测标签
    user_labels_list = list(train_features.keys())
    y_true_list, y_pred_list = [], []

    # ── 匿名化标签 ───────────
    label_map = {real: f"user_{i}" for i, real in enumerate(user_labels_list)}
    # ──────────────────────────

    for feat, true_uid in test_data:
        pred_uid, _ = svm_auth.predict(feat)

        # 使用匿名标签替换真实标签
        y_true_list.append(label_map[true_uid])
        y_pred_list.append(label_map[pred_uid])

    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)

    from sklearn.metrics import f1_score as sk_f1
    svm_macro_f1 = float(sk_f1(
        y_true_arr,  # 这里替换为匿名化的真实标签
        y_pred_arr,  # 这里替换为匿名化的预测标签
        labels=list(label_map.values()),  # 使用匿名标签
        average="macro",
        zero_division=0,
    ))

    # 保存混淆矩阵的路径
    cm_save_path = output_dir / "svm_confusion_matrix.png"
    _plot_svm_confusion_matrix(
        y_true=y_true_arr,  # 使用匿名化的真实标签
        y_pred=y_pred_arr,  # 使用匿名化的预测标签
        labels=list(label_map.values()),  # 传递匿名标签
        save_path=str(cm_save_path),
        cv_accuracy=svm_cv_mean,
        cv_std=svm_cv_std,
        macro_f1=svm_macro_f1,  # 计算得到宏 F1 分数
    )

    # ────────────────────────────────────────────
    # Step 6: 打印对比表格
    # ────────────────────────────────────────────
    _print_comparison_table(comparison)

    # ────────────────────────────────────────────
    # Step 7: 可视化对比图
    # ────────────────────────────────────────────
    fig_path = output_dir / 'method_comparison.png'
    _plot_comparison(comparison, fig_path)

    # ────────────────────────────────────────────
    # Step 7.5: SVM 置信云图
    # ────────────────────────────────────────────
    print("\n[SVM] 绘制置信云图 ...")
    _plot_svm_confidence_map(
        features_by_user=features_by_user,
        svm_auth=svm_auth,
        output_dir=output_dir,
    )

    # ────────────────────────────────────────────
    # Step 8: 保存 JSON
    # ────────────────────────────────────────────
    json_path = output_dir / 'comparison_results.json'
    _safe = lambda v: None if (isinstance(v, float) and np.isnan(v)) else v
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {k: {kk: _safe(vv) for kk, vv in vd.items()}
             for k, vd in comparison.items()},
            f, indent=2, ensure_ascii=False
        )
    print(f"\n[对比] 结果已保存: {json_path}")

    return comparison


# ================================================================
# 内部辅助：打印表格
# ================================================================
def _print_comparison_table(comparison: dict):
    eu = comparison['euclidean']
    sv = comparison['svm']

    eer_svm_str = "N/A (闭集)" if np.isnan(sv['eer']) else f"{sv['eer']*100:.2f}%"
    fnr_svm_str = "N/A (闭集)"
    fpr_svm_str = "N/A (闭集)"

    print("\n")
    print("=" * 62)
    print("  方法对比汇总 (Euclidean Distance  vs  SVM-RBF)")
    print("=" * 62)
    print(f"  {'指标':<18} {'欧氏距离':>16} {'SVM (RBF)':>18}")
    print("-" * 62)
    print(f"  {'测试准确率':<17} {eu['accuracy']*100:>14.2f}%"
          f" {sv['accuracy']*100:>16.2f}%")
    print(f"  {'FNR (拒真率)':<17} {eu['fnr']*100:>14.2f}%"
          f" {fnr_svm_str:>18}")
    print(f"  {'FPR (认假率)':<17} {eu['fpr']*100:>14.2f}%"
          f" {fpr_svm_str:>18}")
    print(f"  {'F1 分数':<18} {eu['f1']:>16.4f}"
          f" {sv['f1']:>18.4f}")
    print(f"  {'EER':<18} {eu['eer']*100:>14.2f}%"
          f" {eer_svm_str:>18}")
    print(f"  {'CV 准确率 (5折)':<16} "
          f"{eu['cv_accuracy']*100:>10.2f}%"
          f"±{eu['cv_std'] * 100:.2f}%"
          f" {sv['cv_accuracy'] * 100:>10.2f}%±{sv['cv_std'] * 100:.2f}%")
    print("=" * 62)

    # 胜负判断
    print("\n  📊 关键指标对比:")
    acc_winner = "SVM" if sv['accuracy'] > eu['accuracy'] else "欧氏距离"
    cv_winner = "SVM" if sv['cv_accuracy'] > eu['cv_accuracy'] else "欧氏距离"
    f1_winner = "SVM" if sv['f1'] > eu['f1'] else "欧氏距离"

    acc_delta = abs(sv['accuracy'] - eu['accuracy']) * 100
    cv_delta = abs(sv['cv_accuracy'] - eu['cv_accuracy']) * 100
    f1_delta = abs(sv['f1'] - eu['f1'])

    print(f"    准确率: {acc_winner} 领先 {acc_delta:.2f}%")
    print(f"    CV准确率: {cv_winner} 领先 {cv_delta:.2f}%")
    print(f"    F1分数: {f1_winner} 领先 {f1_delta:.4f}")

    # 目标达成检查
    target = 0.85
    print(f"\n  🎯 目标准确率 {target * 100:.0f}%:")
    for name, acc in [("欧氏距离", eu['accuracy']), ("SVM", sv['accuracy'])]:
        if acc >= target:
            print(f"    ✓ {name}: {acc * 100:.2f}% — 达标!")
        else:
            print(f"    ✗ {name}: {acc * 100:.2f}% — 差距 {(target - acc) * 100:.2f}%")
    print("=" * 62)


# ================================================================
# 内部辅助：可视化对比图
# ================================================================
# ================================================================
# 内部辅助：SVM 置信云图
# ================================================================
def _plot_svm_confidence_map(features_by_user, svm_auth, output_dir):
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from sklearn.decomposition import PCA

    # ── 1. 只取样本数最多的 5 个用户 ──────────────────────────────
    sorted_users = sorted(
        features_by_user.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    TOP_N    = min(5, len(sorted_users))
    selected = sorted_users[:TOP_N]
    user_names = [f"user_{i}" for i in range(TOP_N)]
    print(f"[置信云图] 选用用户(样本数最多前{TOP_N}): {user_names}")

    # ── 2. 整理数据 ────────────────────────────────────────────────
    X, y = [], []
    for i, (uid, feats) in enumerate(selected):
        for feat in feats:
            X.append(feat)
            y.append(i)
    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    BASE_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    user_colors = BASE_COLORS[:TOP_N]

    # ── 3. StandardScaler 归一化（与 svm_auth 训练时一致）─────────
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 4. PCA 降至 2D ─────────────────────────────────────────────
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var   = pca.explained_variance_ratio_ * 100

    # ── 5. 在 2D 空间训练 SVM（可视化专用，与主模型核参数相同）──────
    #   注意：这是"PCA投影空间"的决策边界，用于可视化，不代表原始模型
    from sklearn.svm import SVC
    svm_2d = SVC(kernel='rbf', C=10, gamma='scale',
                 probability=True, random_state=42)
    svm_2d.fit(X_pca, y)
    train_acc_2d = svm_2d.score(X_pca, y)
    print(f"[置信云图] 2D SVM 训练准确率: {train_acc_2d*100:.1f}%")

    # ── 6. 生成网格 ────────────────────────────────────────────────
    pad  = 0.08
    x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
    y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
    x_min = X_pca[:, 0].min() - pad * x_range
    x_max = X_pca[:, 0].max() + pad * x_range
    y_min = X_pca[:, 1].min() - pad * y_range
    y_max = X_pca[:, 1].max() + pad * y_range

    h    = max(x_range, y_range) / 400
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )
    grid  = np.c_[xx.ravel(), yy.ravel()]
    proba = svm_2d.predict_proba(grid)           # (n_grid, TOP_N)

    Z_class = proba.argmax(axis=1).reshape(xx.shape)
    Z_conf  = proba.max(axis=1).reshape(xx.shape)

    # ── 7. 绘图 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor('white')

    # 置信度着色背景
    bg_rgba = np.zeros((*xx.shape, 4))
    for i, color in enumerate(user_colors):
        rgb  = mcolors.to_rgb(color)
        mask = (Z_class == i)
        bg_rgba[mask, 0] = rgb[0]
        bg_rgba[mask, 1] = rgb[1]
        bg_rgba[mask, 2] = rgb[2]
        bg_rgba[mask, 3] = Z_conf[mask] * 0.85

    ax.imshow(
        bg_rgba,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower', aspect='auto', zorder=0,
    )

    # 决策边界白线
    ax.contour(
        xx, yy, Z_class,
        levels=np.arange(-0.5, TOP_N, 1),
        colors='white', linewidths=1.8, alpha=0.95, zorder=1,
    )

    # 置信度等高线
    cs = ax.contour(
        xx, yy, Z_conf,
        levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        colors='dimgrey', linewidths=0.8,
        linestyles='--', alpha=0.7, zorder=2,
    )
    ax.clabel(cs, fmt='%.1f', fontsize=10, inline=True)

    # 散点
    for i, (uid, color) in enumerate(zip(user_names, user_colors)):
        mask = (y == i)
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, edgecolors='white', linewidths=0.9,
            s=75, zorder=5, label=uid,
        )

    # colorbar
    sm = ScalarMappable(cmap='Greys', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Prediction Confidence', fontsize=13)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_title(
        f"SVM Confidence Map — Top {TOP_N} Users (by sample count)\n"
        f"PCA 2D Projection  |  "
        f"PC1: {var[0]:.1f}%    PC2: {var[1]:.1f}%    "
        f"Total: {sum(var):.1f}% variance explained",
        fontsize=16, fontweight='bold', pad=14,
    )
    ax.set_xlabel(f"Principal Component 1  ({var[0]:.1f}% variance)", fontsize=13)
    ax.set_ylabel(f"Principal Component 2  ({var[1]:.1f}% variance)", fontsize=13)
    ax.legend(
        title=f'Top {TOP_N} Users', title_fontsize=14,
        fontsize=12, loc='upper right',
        framealpha=0.92, edgecolor='#CCCCCC', markerscale=1.3,
    )

    info = (
        f"Kernel : RBF\n"
        f"C      : 10\n"
        f"γ      : scale\n"
        f"Users  : {TOP_N}\n"
        f"Samples: {len(X)}\n"
        f"2D Acc : {train_acc_2d*100:.1f}%"
    )
    ax.text(
        0.01, 0.01, info,
        transform=ax.transAxes, fontsize=12, va='bottom',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                  alpha=0.88, edgecolor='#AED6F1'),
        zorder=6,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle=':', alpha=0.3, zorder=0)
    plt.tight_layout()

    save_path = Path(output_dir) / 'svm_confidence_map.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SVM] 置信云图已保存: {save_path}")


def _plot_svm_confusion_matrix(y_true, y_pred, labels, save_path,
                               cv_accuracy, cv_std, macro_f1):
    """画 SVM 混淆矩阵并保存"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))

    # 增大字体和加粗注释
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 14, "weight": "bold"},  # 可以根据需要修改字体大小
        linewidths=0.4, linecolor="lightgrey",
        ax=ax,
    )

    # 对角线加深色边框
    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=False, edgecolor="#1A5276", lw=2.0
        ))

    overall_acc = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0
    ax.set_title(
        f"Confusion Matrix — SVM Identification\n"
        f"Test Accuracy: {overall_acc * 100:.2f}%"
        f"   |   Macro F1: {macro_f1:.4f}",
        fontsize=18, fontweight="bold", pad=16,  # 增大标题字体大小
    )
    ax.set_ylabel("True Label", fontsize=16, fontweight="bold")  # 增大Y轴标签的字体大小
    ax.set_xlabel("Predicted Label", fontsize=16, fontweight="bold")  # 增大X轴标签的字体大小
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14, fontweight="bold")  # 增大X轴刻度的字体大小
    ax.set_yticklabels(labels, rotation=0, fontsize=14, fontweight="bold")  # 增大Y轴刻度的字体大小

    # 右下角准确率标注
    ax.text(
        0.99, 0.01,
        f"Overall Acc: {overall_acc * 100:.2f}%",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=14, fontweight="bold", color="#1A5276",  # 增大文本字体大小
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#D6EAF8", alpha=0.8),
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SVM] 混淆矩阵已保存: {save_path}")


# 示例数据
# 将 y_true 和 y_pred 替换为您的数据
num_users = 10  # 假设有 10 个用户
y_true = np.random.choice([f'user{i}' for i in range(num_users)], size=100)
y_pred = np.random.choice([f'user{i}' for i in range(num_users)], size=100)

# 生成标签
labels = [f'user{i}' for i in range(num_users)]

# 保存路径
save_path = 'output/confusion_matrix.png'

# 示例的交叉验证精度、标准差和宏 F1 分数
cv_accuracy = 0.85
cv_std = 0.03
macro_f1 = 0.80

# 调用绘图函数
_plot_svm_confusion_matrix(y_true, y_pred, labels, save_path, cv_accuracy, cv_std, macro_f1)

def _plot_comparison(comparison: dict, save_path: Path):
    """
    Generate a 4-subplot method comparison visualization:
      - Bar chart: Accuracy / F1
      - Radar chart: Multi-dimensional comparison
      - Error bar chart: CV Accuracy ± std
      - Summary table
    """
    eu = comparison['euclidean']
    sv = comparison['svm']

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('BreathSign — Method Comparison: Euclidean Distance vs SVM (RBF)',
                 fontsize=15, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.28, wspace=0.15)          # ← wspace 缩小

    colors = {'euclidean': '#4C72B0', 'svm': '#DD8452'}
    labels = ['Euclidean', 'SVM (RBF)']

    # ── Subplot 1: Bar Chart (Accuracy / F1 / CV Accuracy) ──────
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['Test Accuracy', 'F1 Score', 'CV Accuracy']
    eu_vals = [eu['accuracy'], eu['f1'], eu['cv_accuracy']]
    sv_vals = [sv['accuracy'], sv['f1'], sv['cv_accuracy']]

    x = np.arange(len(metrics_names))
    width = 0.32

    bars_eu = ax1.bar(x - width / 2, eu_vals, width,
                      label='Euclidean', color=colors['euclidean'],
                      alpha=0.85, edgecolor='white', linewidth=0.8)
    bars_sv = ax1.bar(x + width / 2, sv_vals, width,
                      label='SVM (RBF)', color=colors['svm'],
                      alpha=0.85, edgecolor='white', linewidth=0.8)

    for bar in bars_eu:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f'{h * 100:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars_sv:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f'{h * 100:.1f}%', ha='center', va='bottom', fontsize=8)

    ax1.axhline(0.85, color='red', linestyle='--', linewidth=1.2,
                label='Target 85%')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('Core Metrics Comparison', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax1.set_axisbelow(True)

    # ── Subplot 2: Radar Chart (5 dimensions) ───────────────────
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    radar_labels = ['Test Acc', 'F1', 'CV Acc', '1-FNR', '1-FPR']
    eu_radar = [
        eu['accuracy'],
        eu['f1'],
        eu['cv_accuracy'],
        1 - eu['fnr'],
        1 - eu['fpr'],
    ]
    sv_radar = [
        sv['accuracy'],
        sv['f1'],
        sv['cv_accuracy'],
        sv['accuracy'],   # substitute for 1-FNR
        sv['accuracy'],   # substitute for 1-FPR
    ]

    N = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    eu_radar += eu_radar[:1]
    sv_radar += sv_radar[:1]

    ax2.plot(angles, eu_radar, 'o-', linewidth=2,
             color=colors['euclidean'], label='Euclidean')
    ax2.fill(angles, eu_radar, alpha=0.20, color=colors['euclidean'])
    ax2.plot(angles, sv_radar, 's-', linewidth=2,
             color=colors['svm'], label='SVM (RBF)')
    ax2.fill(angles, sv_radar, alpha=0.20, color=colors['svm'])

    ax2.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)  # ← 调大
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)   # ← 调大
    ax2.set_title('Multi-dimensional Radar\n(SVM FNR/FPR replaced by Accuracy)',
                  fontsize=10, fontweight='bold', pad=14)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # ── Subplot 3: CV Accuracy ± std Error Bar ───────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    method_x = [0, 1]
    cv_accs = [eu['cv_accuracy'], sv['cv_accuracy']]
    cv_stds = [eu['cv_std'],      sv['cv_std']]
    bar_colors = [colors['euclidean'], colors['svm']]

    bars = ax3.bar(method_x, cv_accs, 0.45,
                   color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.8)
    ax3.errorbar(method_x, cv_accs, yerr=cv_stds,
                 fmt='none', ecolor='black',
                 elinewidth=2, capsize=8, capthick=2)

    for bar, acc, std in zip(bars, cv_accs, cv_stds):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 acc + std + 0.015,
                 f'{acc * 100:.2f}%\n±{std * 100:.2f}%',
                 ha='center', va='bottom', fontsize=9,
                 fontweight='bold')

    ax3.axhline(0.85, color='red', linestyle='--',
                linewidth=1.2, label='Target 85%')
    ax3.set_xticks(method_x)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_ylim(0, min(1.0, max(cv_accs) + max(cv_stds) + 0.15))
    ax3.set_ylabel('CV Accuracy', fontsize=10)
    ax3.set_title('5-Fold CV Accuracy ± Std Dev', fontsize=11,
                  fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax3.set_axisbelow(True)

    # ── Subplot 4: Summary Table ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    try:
        eer_svm_val = float(sv['eer'])
        eer_svm_str = "N/A" if np.isnan(eer_svm_val) else f"{eer_svm_val * 100:.2f}%"
    except (ValueError, TypeError):
        eer_svm_str = "N/A"
        eer_svm_val = float('nan')

    table_data = [
        ['Metric',        'Euclidean',        'SVM (RBF)',      'Winner'],
        ['Test Accuracy',
         f"{eu['accuracy'] * 100:.2f}%",
         f"{sv['accuracy'] * 100:.2f}%",
         'SVM' if sv['accuracy'] >= eu['accuracy'] else 'Euclidean'],
        ['F1 Score',
         f"{eu['f1']:.4f}",
         f"{sv['f1']:.4f}",
         'SVM' if sv['f1'] >= eu['f1'] else 'Euclidean'],
        ['EER',
         f"{eu['eer'] * 100:.2f}%",
         eer_svm_str,
         'Euclidean' if np.isnan(eer_svm_val)
         else ('SVM' if eer_svm_val <= eu['eer'] else 'Euclidean')],
        ['FNR',
         f"{eu['fnr'] * 100:.2f}%",
         'N/A (closed)',
         '—'],
        ['FPR',
         f"{eu['fpr'] * 100:.2f}%",
         'N/A (closed)',
         '—'],
        ['CV Accuracy',
         f"{eu['cv_accuracy'] * 100:.2f}%\n±{eu['cv_std'] * 100:.2f}%",
         f"{sv['cv_accuracy'] * 100:.2f}%\n±{sv['cv_std'] * 100:.2f}%",
         'SVM' if sv['cv_accuracy'] >= eu['cv_accuracy'] else 'Euclidean'],
        ['Pass (≥85%)',
         '✓' if eu['accuracy'] >= 0.85 else '✗',
         '✓' if sv['accuracy'] >= 0.85 else '✗',
         ''],
    ]

    tbl = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.12, 1.68)

    # Header style
    for j in range(4):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row colors + winner column highlight
    row_colors = ['#EBF5FB', '#FDFEFE']
    for i in range(1, len(table_data)):
        for j in range(4):
            tbl[i, j].set_facecolor(row_colors[(i - 1) % 2])
        win_cell = tbl[i, 3]
        if 'SVM' in str(table_data[i][3]):
            win_cell.set_facecolor('#FDEBD0')
        elif 'Euclidean' in str(table_data[i][3]):
            win_cell.set_facecolor('#D5F5E3')

    ax4.set_title('Detailed Metrics Summary', fontsize=11,
                  fontweight='bold', pad=10)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"[Comparison] Visualization saved: {save_path}")




# ================================================================
# train_system —— 在原有流程末尾插入 compare_methods
# ================================================================
# ================================================================
# train_system —— 支持从 JSON 加载 或 从音频目录提取
# ================================================================
def train_system(data_dir_or_json, output_dir, test_ratio=0.2):
    """
    训练系统
    - 若 data_dir_or_json 以 .json 结尾，直接读取特征
    - 否则按原逻辑从音频目录提取特征
    """
    print("=" * 60)
    print("BreathSign 系统训练")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    system = BreathSignSystem()

    # ──────────────────────────────────────────────
    # 阶段1: 特征提取 或 从 JSON 加载
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)

    if str(data_dir_or_json).endswith('.json'):
        # ── 从 JSON 直接加载 ──────────────────────
        print("阶段1: 从 JSON 加载特征")
        print("-" * 40)
        features_by_user = load_features_from_json(str(data_dir_or_json))

    else:
        # ── 从音频目录提取（原逻辑，不动） ────────
        print("阶段1: 特征提取 (BCD + RTD)")
        print("-" * 40)

        data_dir = Path(data_dir_or_json)
        users = [d for d in data_dir.iterdir()
                 if d.is_dir() and not d.name.startswith('.')]
        print(f"\n发现 {len(users)} 个用户")

        features_by_user = {}
        for user_dir in sorted(users):
            user_id = user_dir.name
            print(f"\n处理用户: {user_id}")

            wav_files = (list(user_dir.glob('*.wav')) +
                         list(user_dir.glob('*.WAV')))
            print(f"  音频文件数: {len(wav_files)}")

            user_features = []
            for wav_file in wav_files:
                try:
                    result = system.process_audio(str(wav_file))
                    user_features.append(result['final_features'])
                except Exception as e:
                    print(f"    Warning: {wav_file.name} - {e}")

            if len(user_features) > 0:
                max_dim = max(len(f) for f in user_features)
                unified = []
                for f in user_features:
                    if len(f) < max_dim:
                        padded = np.zeros(max_dim)
                        padded[:len(f)] = f
                        unified.append(padded)
                    else:
                        unified.append(f)
                features_by_user[user_id] = np.array(unified)
                print(f"  提取样本数: {len(unified)}, 特征维度: {max_dim}")

    # ──────────────────────────────────────────────
    # 阶段2: 统一特征维度
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段2: 统一特征维度")
    print("-" * 40)

    target_dim = max(f.shape[1] for f in features_by_user.values())
    print(f"  目标维度: {target_dim}")

    for user_id in features_by_user:
        features = features_by_user[user_id]
        if features.shape[1] < target_dim:
            padded = np.zeros((features.shape[0], target_dim))
            padded[:, :features.shape[1]] = features
            features_by_user[user_id] = padded

    # ──────────────────────────────────────────────
    # 阶段3: 划分训练/测试集
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段3: 划分训练/测试集")
    print("-" * 40)

    rng = np.random.RandomState(42)
    train_features = {}
    test_data = []

    for user_id, features in features_by_user.items():
        n_samples = len(features)
        n_test = max(1, int(n_samples * test_ratio))
        if n_samples - n_test < 1:
            n_test = max(0, n_samples - 1)

        indices = rng.permutation(n_samples)
        test_indices  = indices[:n_test]
        train_indices = indices[n_test:]

        if len(train_indices) > 0:
            train_features[user_id] = features[train_indices]
        for idx in test_indices:
            test_data.append((features[idx], user_id))

    print(f"  训练集用户数: {len(train_features)}")
    print(f"  测试集样本数: {len(test_data)}")

    # ──────────────────────────────────────────────
    # 阶段4: 用户注册（欧氏距离方法）
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段4: 用户注册 (欧氏距离)")
    print("-" * 40)

    for user_id, features in train_features.items():
        system.authenticator.enroll(user_id, features)
        print(f"  {user_id}: {len(features)} 样本")

    # ──────────────────────────────────────────────
    # 阶段5: 阈值优化
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段5: 阈值优化")
    print("-" * 40)

    optimal_threshold, eer = system.authenticator.optimize_threshold(test_data)
    system.authenticator.set_threshold(optimal_threshold)
    print(f"  最优阈值: {optimal_threshold:.4f}")
    print(f"  EER:      {eer * 100:.2f}%")

    # ──────────────────────────────────────────────
    # 阶段6: 欧氏距离方法性能评估
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段6: 性能评估 (欧氏距离)")
    print("-" * 40)

    metrics_euclidean = system.authenticator.evaluate(test_data)
    print(f"  测试准确率:  {metrics_euclidean['accuracy'] * 100:.2f}%")
    print(f"  FNR (拒真率):{metrics_euclidean['fnr'] * 100:.2f}%")
    print(f"  FPR (认假率):{metrics_euclidean['fpr'] * 100:.2f}%")
    print(f"  F1分数:      {metrics_euclidean['f1']:.4f}")

    print("\n  交叉验证 (欧氏距离):")
    cv_results, cv_mean_euclidean = cross_validate(features_by_user, n_folds=5)
    print(f"    平均准确率: {cv_mean_euclidean['accuracy'] * 100:.2f}%"
          f" ± {cv_mean_euclidean['accuracy_std'] * 100:.2f}%")
    print(f"    平均FNR:    {cv_mean_euclidean['fnr'] * 100:.2f}%")
    print(f"    平均FPR:    {cv_mean_euclidean['fpr'] * 100:.2f}%")

    # ──────────────────────────────────────────────
    # 阶段7: 保存欧氏距离模型
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段7: 保存欧氏距离模型")
    print("-" * 40)

    model_dir = output_dir / 'model'
    system.save(str(model_dir))
    print(f"  模型目录: {model_dir}")

    # ──────────────────────────────────────────────
    # 阶段8: 方法对比（SVM 训练 + 双方法可视化）
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段8: 方法对比 (欧氏距离 vs SVM)")
    print("-" * 40)

    comparison = compare_methods(
        features_by_user=features_by_user,
        train_features=train_features,
        test_data=test_data,
        output_dir=output_dir,
        eer_euclidean=eer,
        metrics_euclidean=metrics_euclidean,
        cv_mean_euclidean=cv_mean_euclidean,
    )

    # ──────────────────────────────────────────────
    # 汇总保存 training_results.json
    # ──────────────────────────────────────────────
    results = {
        'n_users': len(features_by_user),
        'n_samples': sum(len(f) for f in features_by_user.values()),
        'feature_dim': target_dim,
        'euclidean': {
            'optimal_threshold': float(optimal_threshold),
            'eer': float(eer),
            'test_metrics': {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in metrics_euclidean.items()
                if k != 'predictions'
            },
            'cv_metrics': {k: float(v) for k, v in cv_mean_euclidean.items()},
        },
        'svm': {
            'cv_accuracy':    float(comparison['svm']['cv_accuracy']),
            'cv_std':         float(comparison['svm']['cv_std']),
            'test_accuracy':  float(comparison['svm']['accuracy']),
            'f1':             float(comparison['svm']['f1']),
        },
    }

    with open(output_dir / 'training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────
    # 最终汇总打印
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"\n结果汇总:")
    print(f"  用户数:                {len(features_by_user)}")
    print(f"  总样本数:              {sum(len(f) for f in features_by_user.values())}")
    print(f"  特征维度:              {target_dim}")
    print(f"  [欧氏距离] 测试准确率: {metrics_euclidean['accuracy'] * 100:.2f}%")
    print(f"  [欧氏距离] EER:        {eer * 100:.2f}%")
    print(f"  [SVM]      测试准确率: {comparison['svm']['accuracy'] * 100:.2f}%")
    print(f"  [SVM]      CV 准确率:  {comparison['svm']['cv_accuracy'] * 100:.2f}%"
          f" ± {comparison['svm']['cv_std'] * 100:.2f}%")
    print(f"\n  对比图:   {output_dir / 'method_comparison.png'}")
    print(f"  对比JSON: {output_dir / 'comparison_results.json'}")

    return system, results


# ================================================================
# test_system（不动）
# ================================================================
def test_system(model_dir, test_dir):
    """
    测试系统

    Args:
        model_dir: 模型目录
        test_dir:  测试数据目录
    """
    print("=" * 60)
    print("BreathSign 系统测试")
    print("=" * 60)

    system = BreathSignSystem()
    system.load(model_dir)
    print(f"已加载模型，注册用户: {system.authenticator.get_enrolled_users()}")

    test_dir = Path(test_dir)
    test_data = []

    for user_dir in test_dir.iterdir():
        if not user_dir.is_dir() or user_dir.name.startswith('.'):
            continue

        user_id = user_dir.name
        wav_files = list(user_dir.glob('*.wav')) + list(user_dir.glob('*.WAV'))

        for wav_file in wav_files:
            try:
                result = system.process_audio(str(wav_file))
                test_data.append((result['final_features'], user_id))
            except Exception as e:
                print(f"Warning: {wav_file} - {e}")

    metrics = system.authenticator.evaluate(test_data)
    print(f"\n测试结果:")
    print(f"  样本数: {len(test_data)}")
    print(f"  准确率: {metrics['accuracy'] * 100:.2f}%")
    print(f"  FNR:    {metrics['fnr'] * 100:.2f}%")
    print(f"  FPR:    {metrics['fpr'] * 100:.2f}%")

    return metrics


# ================================================================
# demo（不动）
# ================================================================
def demo(audio_file, model_dir=None):
    """
    演示模式

    Args:
        audio_file: 音频文件路径
        model_dir:  模型目录 (可选)
    """
    print("=" * 60)
    print("BreathSign 演示")
    print("=" * 60)

    system = BreathSignSystem()

    if model_dir:
        system.load(model_dir)
        print(f"已加载模型，注册用户: {system.authenticator.get_enrolled_users()}")

    print(f"\n处理音频: {audio_file}")
    result = system.process_audio(audio_file)

    print(f"\n处理结果:")
    print(f"  采样率:    {result['sr']} Hz")
    print(f"  时长:      {result['duration']:.2f}s")
    print(f"  双通道:    {result['is_stereo']}")
    print(f"  呼吸事件数:{result['n_breath_events']}")
    print(f"  呼吸周期数:{result['n_breath_cycles']}")
    print(f"  特征维度:  {len(result['final_features'])}")

    if len(system.authenticator.get_enrolled_users()) > 0:
        auth_result = system.authenticator.authenticate(result['final_features'])
        print(f"\n认证结果:")
        print(f"  认证状态:  {'通过' if auth_result['authenticated'] else '拒绝'}")
        print(f"  预测用户:  {auth_result['predicted_user']}")
        print(f"  最小距离:  {auth_result['min_distance']:.4f}")

    figure_dir = Path(audio_file).parent
    if result['detection_result']:
        fig_path = figure_dir / f"{Path(audio_file).stem}_analysis.png"
        visualize_detection(result['left_denoised'], result['sr'],
                            result['detection_result'], str(fig_path))
        print(f"\n可视化已保存: {fig_path}")

    return result


# ================================================================
# main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='BreathSign 呼吸声身份认证系统')
    parser.add_argument('--mode', type=str,
                        choices=['train', 'test', 'demo'],
                        default='demo', help='运行模式')
    parser.add_argument('--skip_extract', action='store_true',
                        default=True,  # ← 加这个，默认就跳过提取
                        help='跳过特征提取，直接从 features_by_user.json 加载')
    parser.add_argument('--json_path', type=str,
                        default='/Users/yaojinru/Desktop/breathsign_denoise 3/output/features_by_user.json',
                        # ← 确认这个路径对
                        help='features_by_user.json 路径')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/yaojinru/Desktop/breathsign_denoise 3/results',  # ← 加这个默认输出路径
                        help='输出目录')
    parser.add_argument('--data_dir',   type=str, help='数据目录')
    parser.add_argument('--model_dir',  type=str, help='模型目录')
    parser.add_argument('--audio',      type=str, help='音频文件路径 (demo模式)')

    args = parser.parse_args()
    print(f"运行模式: {args.mode}")

    if args.mode == 'train':
        if not args.output_dir:
            args.output_dir = str(Path(__file__).parent / 'results')

        if args.skip_extract:
            # ── 直接从 JSON 加载特征，跳过音频提取 ──
            print(f"[跳过特征提取] 从 JSON 加载: {args.json_path}")
            train_system(args.json_path, args.output_dir)
        else:
            # ── 原有逻辑：从音频目录提取特征 ──
            if not args.data_dir:
                args.data_dir = str(Path(__file__).parent / 'output')
            train_system(args.data_dir, args.output_dir)

    elif args.mode == 'test':
        if not args.model_dir or not args.data_dir:
            print("Error: --model_dir and --data_dir required for test mode")
            return
        test_system(args.model_dir, args.data_dir)

    elif args.mode == 'demo':
        if args.audio:
            demo(args.audio, args.model_dir)
        else:
            output_dir = Path(__file__).parent / 'output'
            subjects   = [d for d in output_dir.iterdir() if d.is_dir()]
            if subjects:
                wav_files = (list(subjects[0].glob('*.wav')) +
                             list(subjects[0].glob('*.WAV')))
                if wav_files:
                    demo(str(wav_files[0]), args.model_dir)
                else:
                    print("No audio files found for demo")
            else:
                print("No data found for demo. Please run denoise.py first.")


if __name__ == '__main__':
    main()
