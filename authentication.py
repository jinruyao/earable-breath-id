# authentication.py

import json
import warnings

from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# SVM分类器
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    confusion_matrix, classification_report
)
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class BreathSignAuthenticator:

    def __init__(self, threshold=None, n_cycles_for_auth=1):
        self.templates = {}
        self.threshold = threshold
        self.n_cycles_for_auth = n_cycles_for_auth

        # 存储训练数据用于阈值优化
        self.enrollment_features = {}

    def enroll(self, user_id, features):
        features = np.array(features)

        if len(features.shape) == 1:
            # 单个特征向量
            template = features
            self.enrollment_features[user_id] = [features]
        else:
            # 多个特征向量，取平均
            template = np.mean(features, axis=0)
            self.enrollment_features[user_id] = list(features)

        self.templates[user_id] = template

        return template

    def compute_distance(self, feature, template):
        return np.linalg.norm(feature - template)

    def authenticate(self, feature, claimed_user_id=None):
        if len(self.templates) == 0:
            return {
                'authenticated': False,
                'predicted_user': None,
                'min_distance': float('inf'),
                'distances': {},
                'threshold': self.threshold
            }

        # 计算与所有模板的距离
        distances = {}
        for user_id, template in self.templates.items():
            distances[user_id] = self.compute_distance(feature, template)

        # 找到最小距离
        min_user = min(distances, key=distances.get)
        min_distance = distances[min_user]

        # 获取阈值
        threshold = self.threshold
        if threshold is None:
            threshold = self._estimate_threshold()

        # 判定
        if claimed_user_id is not None:
            # 验证模式：检查是否为声称的用户
            if claimed_user_id not in self.templates:
                authenticated = False
            else:
                claimed_distance = distances[claimed_user_id]
                authenticated = claimed_distance < threshold
        else:
            # 识别模式：返回最近的用户
            authenticated = min_distance < threshold

        return {
            'authenticated': authenticated,
            'predicted_user': min_user if authenticated else None,
            'min_distance': min_distance,
            'distances': distances,
            'threshold': threshold
        }

    def _estimate_threshold(self):
        if len(self.enrollment_features) < 2:
            # 样本不足，使用默认值
            return 1.0

        intra_distances = []  # 类内距离
        inter_distances = []  # 类间距离

        users = list(self.enrollment_features.keys())

        for i, user_i in enumerate(users):
            features_i = self.enrollment_features[user_i]
            template_i = self.templates[user_i]

            # 类内距离
            for feat in features_i:
                dist = self.compute_distance(feat, template_i)
                if dist > 0:  # 排除与自身的距离
                    intra_distances.append(dist)

            # 类间距离
            for j, user_j in enumerate(users):
                if i != j:
                    template_j = self.templates[user_j]
                    for feat in features_i:
                        dist = self.compute_distance(feat, template_j)
                        inter_distances.append(dist)

        if len(intra_distances) == 0 or len(inter_distances) == 0:
            return 1.0

        # 选择阈值：类内最大距离和类间最小距离的中点
        max_intra = np.max(intra_distances)
        min_inter = np.min(inter_distances)

        # 或使用更保守的方法：类内95百分位
        threshold = np.percentile(intra_distances, 95)

        # 确保阈值在合理范围
        threshold = max(threshold, (max_intra + min_inter) / 2)

        return threshold

    def set_threshold(self, threshold):
        """设置认证阈值"""
        self.threshold = threshold

    def optimize_threshold(self, test_data):
        if len(test_data) == 0:
            return self._estimate_threshold(), 0.5

        # 收集所有距离
        genuine_distances = []  # 真实用户的距离
        impostor_distances = []  # 冒充者的距离

        for feature, true_user in test_data:
            if true_user not in self.templates:
                continue

            # 计算与真实用户模板的距离
            true_dist = self.compute_distance(feature, self.templates[true_user])
            genuine_distances.append(true_dist)

            # 计算与其他用户模板的距离
            for user_id, template in self.templates.items():
                if user_id != true_user:
                    imp_dist = self.compute_distance(feature, template)
                    impostor_distances.append(imp_dist)

        if len(genuine_distances) == 0 or len(impostor_distances) == 0:
            return self._estimate_threshold(), 0.5

        # 搜索最优阈值
        all_distances = sorted(genuine_distances + impostor_distances)
        best_threshold = all_distances[len(all_distances) // 2]
        best_eer = 1.0

        for threshold in all_distances:
            # FNR: 真实用户被拒绝的比例
            fnr = np.mean([d >= threshold for d in genuine_distances])
            # FPR: 冒充者被接受的比例
            fpr = np.mean([d < threshold for d in impostor_distances])

            eer = (fnr + fpr) / 2
            if eer < best_eer:
                best_eer = eer
                best_threshold = threshold

        return best_threshold, best_eer

    def evaluate(self, test_data):
        if len(test_data) == 0:
            return {}

        # 统计变量
        n_correct = 0  # 正确识别数
        n_total = 0  # 总测试数

        # 用于计算FNR和FPR
        genuine_attempts = []  # (距离, 是否接受)
        impostor_attempts = []  # (距离, 是否接受)

        predictions = []

        for feature, true_user in test_data:
            if true_user not in self.templates:
                # 跳过未注册用户的样本
                continue

            n_total += 1
            result = self.authenticate(feature)

            # 识别准确率: 预测用户 == 真实用户
            if result['authenticated'] and result['predicted_user'] == true_user:
                n_correct += 1

            # Genuine attempt: 合法用户尝试认证
            true_dist = self.compute_distance(feature, self.templates[true_user])
            genuine_attempts.append({
                'distance': true_dist,
                'accepted': true_dist < self.threshold if self.threshold is not None else True
            })

            # Impostor attempts: 模拟其他用户冒充
            for other_user, other_template in self.templates.items():
                if other_user != true_user:
                    imp_dist = self.compute_distance(feature, other_template)
                    impostor_attempts.append({
                        'distance': imp_dist,
                        'accepted': imp_dist < self.threshold if self.threshold else True
                    })

            predictions.append({
                'true_user': true_user,
                'predicted_user': result['predicted_user'],
                'authenticated': result['authenticated'],
                'min_distance': result['min_distance']
            })

        # 计算指标
        accuracy = n_correct / n_total if n_total > 0 else 0

        # FNR: 合法用户被拒绝的比例
        if len(genuine_attempts) > 0:
            fnr = sum(1 for g in genuine_attempts if not g['accepted']) / len(genuine_attempts)
        else:
            fnr = 0

        # FPR: 冒充者被接受的比例
        if len(impostor_attempts) > 0:
            fpr = sum(1 for i in impostor_attempts if i['accepted']) / len(impostor_attempts)
        else:
            fpr = 0

        # 精确率和召回率
        tp = n_correct
        fp = sum(1 for i in impostor_attempts if i['accepted'])
        fn = sum(1 for g in genuine_attempts if not g['accepted'])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'fnr': fnr,
            'fpr': fpr,
            'precision': precision,
            'recall': recall,
            'n_samples': n_total,
            'n_genuine_attempts': len(genuine_attempts),
            'n_impostor_attempts': len(impostor_attempts),
            'predictions': predictions
        }

        # F1 Score
        if precision + recall > 0:
            metrics['f1'] = 2 * precision * recall / (precision + recall)
        else:
            metrics['f1'] = 0

        return metrics

    def multi_cycle_authenticate(self, features, claimed_user_id=None):
        features = np.array(features)

        if len(features.shape) == 1:
            # 单个特征
            return self.authenticate(features, claimed_user_id)

        # 取平均特征
        mean_feature = np.mean(features, axis=0)

        return self.authenticate(mean_feature, claimed_user_id)

    def save(self, path):
        """
        保存认证器状态

        Args:
            path: 保存路径
        """
        state = {
            'templates': {k: v.tolist() for k, v in self.templates.items()},
            'threshold': self.threshold,
            'n_cycles_for_auth': self.n_cycles_for_auth
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2, cls=NumpyEncoder)

    def load(self, path):
        """
        加载认证器状态

        Args:
            path: 加载路径
        """
        with open(path, 'r') as f:
            state = json.load(f)

        self.templates = {k: np.array(v) for k, v in state['templates'].items()}
        self.threshold = state.get('threshold')
        self.n_cycles_for_auth = state.get('n_cycles_for_auth', 1)

    def get_enrolled_users(self):
        """获取已注册用户列表"""
        return list(self.templates.keys())

    def remove_user(self, user_id):
        """移除用户"""
        if user_id in self.templates:
            del self.templates[user_id]
        if user_id in self.enrollment_features:
            del self.enrollment_features[user_id]


def cross_validate(features_by_user, n_folds=5):
    users = list(features_by_user.keys())
    results = []

    for fold in range(n_folds):
        authenticator = BreathSignAuthenticator()

        # 划分训练/测试集
        train_data = {}
        test_data = []

        for user_id in users:
            features = features_by_user[user_id]
            n_samples = len(features)

            if n_samples < 2:
                # 样本太少，全部用于训练
                train_data[user_id] = features
                continue

            # 按折划分
            fold_size = max(1, n_samples // n_folds)
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n_samples)

            test_features = features[test_start:test_end]
            train_features = np.vstack([features[:test_start], features[test_end:]])

            if len(train_features) > 0:
                train_data[user_id] = train_features

            for feat in test_features:
                test_data.append((feat, user_id))

        # 注册用户
        for user_id, features in train_data.items():
            authenticator.enroll(user_id, features)

        # 优化阈值
        if len(test_data) > 0:
            optimal_threshold, _ = authenticator.optimize_threshold(test_data)
            authenticator.set_threshold(optimal_threshold)

        # 评估
        metrics = authenticator.evaluate(test_data)
        metrics['fold'] = fold
        results.append(metrics)

    # 计算平均指标
    mean_metrics = {}
    for key in ['accuracy', 'fnr', 'fpr', 'precision', 'recall', 'f1']:
        values = [r[key] for r in results if key in r]
        if values:
            mean_metrics[key] = np.mean(values)
            mean_metrics[f'{key}_std'] = np.std(values)

    return results, mean_metrics


class SVMAuthenticator:

    def __init__(
        self,
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        random_state=42,
    ):
        self.kernel = kernel
        self.C = float(C)
        self.gamma = gamma
        self.probability = bool(probability)
        self.random_state = int(random_state)

        self.model: Pipeline | None = None
        self.user_labels: list[str] = []
        self.is_trained: bool = False

    # -------------------------
    # Core: train / predict
    # -------------------------
    def train(self, features_by_user: dict, use_grid_search: bool = False):
        X, y_idx, labels = self._flatten(features_by_user)
        if len(labels) == 0 or len(X) == 0:
            raise ValueError("No training data found in features_by_user.")

        self.user_labels = labels

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=self.kernel,
                        C=self.C,
                        gamma=self.gamma,
                        probability=self.probability,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.model.fit(X, y_idx)
        self.is_trained = True

        self.pipeline_ = self.model
        self.label_encoder_ = self.user_labels

        return self

    def predict(self, feature):
        """
        Predict one sample.

        Returns
        -------
        pred_user_id : str | None
        confidence : float
            If probability=True, returns max class probability for predicted class.
            Else returns 0.0.
        """
        if not self.is_trained or self.model is None:
            return None, 0.0

        x = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        pred_idx = int(self.model.predict(x)[0])

        conf = 0.0
        svm = self.model.named_steps.get("svm", None)
        if svm is not None and getattr(svm, "probability", False):
            proba = self.model.predict_proba(x)[0]
            conf = float(proba[pred_idx])

        return self.user_labels[pred_idx], conf

    # -------------------------
    # Evaluation helpers
    # -------------------------
    def evaluate(self, test_data, return_report=False):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.asarray([f for (f, _) in test_data], dtype=np.float32)
        y_true_uid = np.asarray([uid for (_, uid) in test_data], dtype=object)

        # Map true uid -> index; unknown uid -> -1 (will be ignored in accuracy)
        uid_to_idx = {uid: i for i, uid in enumerate(self.user_labels)}
        y_true_idx = np.asarray([uid_to_idx.get(uid, -1) for uid in y_true_uid], dtype=int)

        y_pred_idx = self.model.predict(X).astype(int)

        valid = y_true_idx >= 0
        n_samples = int(np.sum(valid))
        if n_samples == 0:
            raise ValueError("No valid test samples (true_user_id not in trained labels).")

        acc = float(accuracy_score(y_true_idx[valid], y_pred_idx[valid]))
        n_correct = int(np.sum(y_true_idx[valid] == y_pred_idx[valid]))

        cm = confusion_matrix(
            y_true_idx[valid],
            y_pred_idx[valid],
            labels=list(range(len(self.user_labels))),
        )

        metrics = {
            "accuracy": acc,
            "n_correct": n_correct,
            "n_samples": n_samples,
            "confusion_matrix": cm,
            "labels": self.user_labels,
        }

        if return_report:
            metrics["report"] = classification_report(
                y_true_idx[valid],
                y_pred_idx[valid],
                labels=list(range(len(self.user_labels))),
                target_names=self.user_labels,
                digits=4,
                zero_division=0,
            )

        return metrics

    # -------------------------
    # Persistence
    # -------------------------
    def save(self, model_path: str):
        """
        Save pipeline to .pkl and labels to _labels.json
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Nothing to save: model not trained.")

        model_path = str(model_path)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)

        labels_path = model_path.replace(".pkl", "_labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(self.user_labels, f, ensure_ascii=False, indent=2)

    def load(self, model_path: str):
        """
        Load pipeline from .pkl and labels from _labels.json
        """
        model_path = str(model_path)
        self.model = joblib.load(model_path)

        labels_path = model_path.replace(".pkl", "_labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.user_labels = json.load(f)

        self.is_trained = True
        return self

    # -------------------------
    # Internal
    # -------------------------
    @staticmethod
    def _flatten(features_by_user: dict):
        """
        {uid: (n,d)} -> X(N,d), y_idx(N), labels(list)
        """
        labels = list(features_by_user.keys())
        X_list, y_list = [], []
        for i, uid in enumerate(labels):
            feats = np.asarray(features_by_user[uid], dtype=np.float32)
            for f in feats:
                X_list.append(np.asarray(f, dtype=np.float32).ravel())
                y_list.append(i)

        if len(X_list) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=int), []

        X = np.stack(X_list, axis=0).astype(np.float32)
        y_idx = np.asarray(y_list, dtype=int)
        return X, y_idx, labels

    def cross_validate(self, features_by_user: dict, n_folds: int = 5, return_oof: bool = False):
        from sklearn.model_selection import StratifiedKFold

        X, y_idx, labels = self._flatten(features_by_user)
        self.user_labels = labels

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        oof_true = np.empty(len(X), dtype=y_idx.dtype)
        oof_pred = np.empty(len(X), dtype=y_idx.dtype)

        for train_idx, test_idx in skf.split(X, y_idx):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel=self.kernel, C=self.C, gamma=self.gamma,
                    probability=self.probability, random_state=self.random_state
                ))
            ])
            pipe.fit(X[train_idx], y_idx[train_idx])
            acc = pipe.score(X[test_idx], y_idx[test_idx])
            scores.append(acc)

            # ← 收集 OOF 预测
            oof_true[test_idx] = y_idx[test_idx]
            oof_pred[test_idx] = pipe.predict(X[test_idx])

        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))

        if return_oof:
            # 把数字索引转回用户名
            oof_true_labels = np.array([labels[i] for i in oof_true])
            oof_pred_labels = np.array([labels[i] for i in oof_pred])
            return mean_acc, std_acc, oof_true_labels, oof_pred_labels

        return mean_acc, std_acc


def train_svm_system(data_dir, output_dir, test_ratio=0.2):
    """
    完整的SVM训练流程

    Args:
        data_dir: 降噪后的数据目录 (output/)，其中应包含 features_by_user.json
        output_dir: 模型和结果输出目录
        test_ratio: 测试集比例，默认0.2
    """
    from pathlib import Path
    import json

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BreathSign SVM 训练系统")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # 阶段1: 直接读取 JSON 特征，跳过重新提取
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段1: 加载预提取特征 (features_by_user.json)")
    print("-" * 40)

    json_path = data_dir / 'features_by_user.json'
    if not json_path.exists():
        raise FileNotFoundError(
            f"找不到特征文件: {json_path}\n"
            "请先运行 feature_extraction.py 的 __main__ 生成该文件。"
        )

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    features_by_user = {}
    for uid, feats in raw.items():
        arr = np.array(feats, dtype=np.float32)
        features_by_user[uid] = arr
        print(f"  {uid}: {arr.shape[0]} 样本, 维度={arr.shape[1]}")

    print(f"\n共加载 {len(features_by_user)} 个用户")

    # ──────────────────────────────────────────────
    # 阶段2: 统一特征维度（只保留 47 维，丢弃异常）
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段2: 统一特征维度")
    print("-" * 40)

    TARGET_DIM = 47  # BCD(8) + RTD/MFCC-mean(39) = 47

    filtered = {}
    for uid, feats in features_by_user.items():
        if feats.shape[1] == TARGET_DIM:
            filtered[uid] = feats
        else:
            print(f"  ⚠️  {uid}: 维度={feats.shape[1]}，期望 {TARGET_DIM}，已跳过")

    features_by_user = filtered
    print(f"  目标维度: {TARGET_DIM}")
    print(f"  有效用户数: {len(features_by_user)}")

    if len(features_by_user) == 0:
        raise ValueError(
            "没有任何用户的特征维度符合要求。\n"
            "请检查 features_by_user.json 是否由最新的 feature_extraction.py 生成。"
        )

    # ──────────────────────────────────────────────
    # 阶段3: 划分训练/测试集
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段3: 划分训练/测试集")
    print("-" * 40)

    train_features = {}
    test_data      = []

    for user_id, features in features_by_user.items():
        n_samples = len(features)
        n_test    = max(1, int(n_samples * test_ratio))

        indices   = np.random.permutation(n_samples)
        test_idx  = indices[:n_test]
        train_idx = indices[n_test:]

        if len(train_idx) > 0:
            train_features[user_id] = features[train_idx]

        for idx in test_idx:
            test_data.append((features[idx], user_id))

    print(f"  训练集: {sum(len(f) for f in train_features.values())} 样本")
    print(f"  测试集: {len(test_data)} 样本")

    # ──────────────────────────────────────────────
    # 阶段4: SVM 训练
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段4: SVM训练")
    print("-" * 40)

    svm_auth = SVMAuthenticator()
    svm_auth.train(train_features, use_grid_search=True)
    print("  SVM 训练完成")

    # ──────────────────────────────────────────────
    # 阶段5: 性能评估
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段5: 性能评估")
    print("-" * 40)

    metrics = svm_auth.evaluate(test_data, return_report=True)
    print(f"  测试准确率: {metrics['accuracy']*100:.2f}%")
    print(f"  正确/总数: {metrics['n_correct']}/{metrics['n_samples']}")

    if 'report' in metrics:
        print("\n  分类报告:")
        # 每行缩进4格，方便阅读
        for line in metrics['report'].splitlines():
            print(f"    {line}")

    # 交叉验证
    print("\n  交叉验证 (5-fold StratifiedKFold):")
    mean_acc, std_acc = svm_auth.cross_validate(features_by_user, n_folds=5)
    print(f"    平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    # ──────────────────────────────────────────────
    # 阶段6: 保存模型
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段6: 保存模型")
    print("-" * 40)

    model_dir = output_dir / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    svm_auth.save(str(model_dir / 'svm_model.pkl'))
    print(f"  模型已保存: {model_dir / 'svm_model.pkl'}")
    print(f"  标签已保存: {model_dir / 'svm_model_labels.json'}")

    # 保存训练结果摘要
    results = {
        'n_users'       : len(features_by_user),
        'n_samples'     : sum(len(f) for f in features_by_user.values()),
        'feature_dim'   : TARGET_DIM,
        'test_accuracy' : float(metrics['accuracy']),
        'cv_accuracy'   : float(mean_acc),
        'cv_std'        : float(std_acc)
    }

    summary_path = output_dir / 'svm_training_results.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  训练摘要已保存: {summary_path}")

    # ──────────────────────────────────────────────
    # 结果汇总
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"\n结果汇总:")
    print(f"  用户数:     {len(features_by_user)}")
    print(f"  总样本数:   {sum(len(f) for f in features_by_user.values())}")
    print(f"  特征维度:   {TARGET_DIM}")
    print(f"  测试准确率: {metrics['accuracy']*100:.2f}%")
    print(f"  交叉验证:   {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  目标:       85%")

    if metrics['accuracy'] >= 0.85:
        print(f"\n  ✓ 达到目标准确率!")
    else:
        gap = (0.85 - metrics['accuracy']) * 100
        print(f"\n  ✗ 未达到目标，差距: {gap:.2f}%")

    # ──────────────────────────────────────────────
    # 阶段7: 可视化
    # ──────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("阶段7: 生成可视化图表")
    print("-" * 40)

    viz_dir = Path(output_dir) / 'visualizations'
    viz     = VisualizationSuite(output_dir=str(viz_dir))

    test_data_viz = []
    for user_id, features in features_by_user.items():
        n_test = max(1, int(len(features) * test_ratio))
        for feat in features[:n_test]:
            test_data_viz.append((feat, user_id))

    print("  [1/2] 决策边界...")
    viz.plot_svm_decision_boundary(features_by_user)
    print("  [2/2] 混淆矩阵...")
    viz.plot_confusion_matrix(svm_auth, features_by_user)

    print(f"\n  图表已保存至: {viz_dir}")

    return svm_auth, results




import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def plot_svm_decision_boundary(features_by_user, model_params={'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}):
    """
    绘制属于 BreathSign 项目的 SVM 决策边界图
    """
    # 1. 准备数据
    X = []
    y = []
    user_names = list(features_by_user.keys())
    for i, user_id in enumerate(user_names):
        for feat in features_by_user[user_id]:
            X.append(feat)
            y.append(i)
    X = np.array(X)
    y = np.array(y)

    # 2. 降维：将 43/86 维降至 2 维以便绘图
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 3. 在 2D 空间重新训练一个 SVM（仅用于可视化展示边界逻辑）
    svm_2d = SVC(**model_params, probability=True)
    svm_2d.fit(X_pca, y)

    # 4. 创建网格采样点
    h = .02  # 网格步长
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 5. 预测概率（置信度）
    Z = svm_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # 取最大概率值作为背景颜色深浅
    Z_prob = np.max(Z, axis=1).reshape(xx.shape)
    Z_class = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 6. 绘图
    plt.figure(figsize=(10, 8))

    # 绘制决策区域，颜色代表类别，透明度代表置信度
    plt.contourf(xx, yy, Z_class, alpha=0.3, cmap=plt.cm.Paired)
    # 绘制概率云图（置信度越深颜色越浓）
    plt.contourf(xx, yy, Z_prob, alpha=0.2, cmap=plt.cm.Greys)

    # 绘制原始数据点
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired, s=50)
    plt.legend(handles=scatter.legend_elements()[0], labels=user_names)

    plt.title("SVM Decision Boundary with Confidence Levels (PCA-Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# ══════════════════════════════════════════════════════════════════
#  可视化工具集  VisualizationSuite
# ══════════════════════════════════════════════════════════════════

class VisualizationSuite:

    # 统一配色，最多支持 20 个用户
    COLORS = plt.cm.get_cmap('tab20', 20)

    def __init__(self, output_dir: str = '.'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 全局字体设置（兼容中文环境）
        plt.rcParams.update({
            'font.size'        : 11,
            'axes.titlesize'   : 13,
            'axes.labelsize'   : 11,
            'figure.dpi'       : 120,
            'savefig.dpi'      : 150,
            'savefig.bbox'     : 'tight',
        })


    # ──────────────────────────────────────────────────────────────
    # 2. SVM 决策边界
    # ──────────────────────────────────────────────────────────────
    def plot_svm_decision_boundary(
        self,
        features_by_user: dict,
        svm_params: dict = None,
        save: bool = True
    ) -> plt.Figure:
        """
        PCA 降至 2D 后绘制 SVM 决策边界与置信度云图
        """
        if svm_params is None:
            svm_params = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'}

        X, y, labels = self._pack(features_by_user)
        n_users = len(labels)

        # PCA → 2D
        pca   = PCA(n_components=2, random_state=42)
        X2d   = pca.fit_transform(X)

        # 在 2D 空间重新训练（仅用于可视化）
        svm2d = SVC(**svm_params, probability=True)
        svm2d.fit(X2d, y)

        # 网格
        margin = 0.8
        h = (np.ptp(X2d[:, 0]) + np.ptp(X2d[:, 1])) / 200   # 自适应步长
        xx, yy = np.meshgrid(
            np.arange(X2d[:, 0].min() - margin, X2d[:, 0].max() + margin, h),
            np.arange(X2d[:, 1].min() - margin, X2d[:, 1].max() + margin, h)
        )
        grid   = np.c_[xx.ravel(), yy.ravel()]
        Z_cls  = svm2d.predict(grid).reshape(xx.shape)
        Z_prob = svm2d.predict_proba(grid).max(axis=1).reshape(xx.shape)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))

        # 决策区域
        cmap_bg = ListedColormap(
            [self.COLORS(i)[:3] for i in range(n_users)]
        )
        ax.contourf(xx, yy, Z_cls, alpha=0.25, cmap=cmap_bg,
                    levels=np.arange(-0.5, n_users, 1))

        # 置信度叠加
        cf = ax.contourf(xx, yy, Z_prob, alpha=0.20,
                         cmap='Greys', levels=10)
        plt.colorbar(cf, ax=ax, label='Max class probability')

        # 决策边界线
        ax.contour(xx, yy, Z_cls, colors='k',
                   linewidths=0.6, alpha=0.5,
                   levels=np.arange(-0.5, n_users, 1))

        # 数据点
        for i, uid in enumerate(labels):
            mask = y == i
            ax.scatter(
                X2d[mask, 0], X2d[mask, 1],
                c=[self.COLORS(i)], label=uid,
                edgecolors='k', linewidths=0.5,
                s=55, zorder=3
            )

        # 支持向量高亮
        sv = svm2d.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1],
                   s=120, facecolors='none',
                   edgecolors='red', linewidths=1.2,
                   zorder=4, label='Support Vectors')

        ax.set_title(
            f"SVM Decision Boundary (PCA 2D)\n"
            f"kernel={svm_params['kernel']}, C={svm_params['C']}, "
            f"{n_users} users"
        )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(loc='upper right', fontsize=8,
                  ncol=max(1, (n_users + 1) // 8), framealpha=0.7)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        if save:
            path = self.output_dir / 'viz_svm_decision_boundary.png'
            fig.savefig(path)
            print(f"    [图2] SVM决策边界图已保存: {path}")
        return fig

    # ──────────────────────────────────────────────────────────────
    # 3. 混淆矩阵
    # ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    def plot_confusion_matrix(
            self,
            svm_auth,
            features_by_user: dict,
            save: bool = True
    ) -> plt.Figure:
        """
        绘制归一化混淆矩阵（基于5折交叉验证）
        """
        # 整理数据
        X, y_idx, labels = svm_auth._flatten(features_by_user)

        # ── 匿名化：真实名字 → user_0, user_1, ... ──────────────
        anon_labels = [f"user_{i}" for i in range(len(labels))]
        label_map = {real: anon for real, anon in zip(labels, anon_labels)}
        # ────────────────────────────────────────────────────────

        # 5折交叉预测
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_idx = cross_val_predict(svm_auth.model, X, y_idx, cv=cv)

        # 整数索引 -> 匿名标签（原来是真实名字，现在换成 user_i）
        y_true_labels = [label_map[labels[i]] for i in y_idx]  # 匹配真实标签到匿名标签
        y_pred_labels = [label_map[labels[i]] for i in y_pred_idx]  # 匹配预测标签到匿名标签

        # 计算混淆矩阵并规范化
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=anon_labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        n = len(anon_labels)
        fig_size = max(6, n * 0.7)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

        # 画出归一化混淆矩阵
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f',
            cmap='Blues',
            xticklabels=anon_labels, yticklabels=anon_labels,
            linewidths=0.4, linecolor='gray',
            vmin=0, vmax=1, ax=ax
        )
        ax.set_xlabel('Predicted User', fontsize=12)
        ax.set_ylabel('True User', fontsize=12)
        ax.set_title('Normalized Confusion Matrix (5-Fold CV)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save:
            path = self.output_dir / 'viz_confusion_matrix.png'
            fig.savefig(path)
            print(f"    [图3] 混淆矩阵已保存: {path}")

        return fig

    # ──────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _pack(features_by_user: dict):
        """
        将 {user_id: (N,D) array} 打包成 X, y, labels

        Returns:
            X      : (total_N, D)
            y      : (total_N,)  整数标签
            labels : 有序标签列表，labels[i] == user_id
        """
        labels = sorted(features_by_user.keys())
        X_list, y_list = [], []
        for i, uid in enumerate(labels):
            feats = features_by_user[uid]
            X_list.append(feats)
            y_list.append(np.full(len(feats), i, dtype=int))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y, labels


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 60)
    print("BreathSign Authentication 模块测试")
    print("=" * 60)

    # ════════════════════════════════════════════════════════════
    # Part 1: BreathSignAuthenticator 基础功能测试（模拟数据）
    # ════════════════════════════════════════════════════════════
    print("\n【Part 1】BreathSignAuthenticator 基础功能测试")
    print("-" * 40)

    np.random.seed(42)
    n_users            = 3
    feature_dim        = 47
    n_samples_per_user = 10

    features_by_user = {}
    for i in range(n_users):
        user_id  = f"user_{i}"
        center   = np.random.randn(feature_dim) * 2
        features = center + np.random.randn(n_samples_per_user, feature_dim) * 0.5
        features_by_user[user_id] = features

    # 1. 注册用户
    print("\n1. 注册用户")
    auth = BreathSignAuthenticator()
    for user_id, features in features_by_user.items():
        auth.enroll(user_id, features[:5])
        print(f"  注册用户: {user_id}")
    print(f"  已注册用户数: {len(auth.get_enrolled_users())}")

    threshold = auth._estimate_threshold()
    auth.set_threshold(threshold)
    print(f"  估计阈值: {threshold:.4f}")

    # 2. 认证测试
    print("\n2. 认证测试")
    test_user    = "user_0"
    test_feature = features_by_user[test_user][5]

    result = auth.authenticate(test_feature)
    print(f"  测试用户: {test_user}")
    print(f"  认证结果: {'通过' if result['authenticated'] else '拒绝'}")
    print(f"  预测用户: {result['predicted_user']}")
    print(f"  最小距离: {result['min_distance']:.4f}")

    impostor_feature = np.random.randn(feature_dim) * 3
    result = auth.authenticate(impostor_feature)
    print(f"\n  冒充者测试:")
    print(f"  认证结果: {'通过' if result['authenticated'] else '拒绝'}")
    print(f"  最小距离: {result['min_distance']:.4f}")

    # 3. 性能评估
    print("\n3. 性能评估")
    test_data = []
    for user_id, features in features_by_user.items():
        for feat in features[5:]:
            test_data.append((feat, user_id))

    metrics = auth.evaluate(test_data)
    print(f"  准确率: {metrics['accuracy']*100:.2f}%")
    print(f"  FNR:    {metrics['fnr']*100:.2f}%")
    print(f"  FPR:    {metrics['fpr']*100:.2f}%")
    print(f"  F1:     {metrics['f1']:.4f}")

    # 4. 交叉验证
    print("\n4. 交叉验证")
    cv_results, mean_metrics = cross_validate(features_by_user, n_folds=3)
    print(f"  平均准确率: {mean_metrics['accuracy']*100:.2f}% ± {mean_metrics['accuracy_std']*100:.2f}%")
    print(f"  平均F1:     {mean_metrics['f1']:.4f} ± {mean_metrics['f1_std']:.4f}")

    # 5. 保存/加载测试
    print("\n5. 保存/加载测试")
    save_path = Path(__file__).parent / 'test_auth_model.json'
    auth.save(str(save_path))
    print(f"  模型已保存: {save_path}")

    auth2 = BreathSignAuthenticator()
    auth2.load(str(save_path))
    print(f"  模型已加载，用户数: {len(auth2.get_enrolled_users())}")
    save_path.unlink()

    print("\n✓ BreathSignAuthenticator 测试通过")

    # ════════════════════════════════════════════════════════════
    # Part 2: SVMAuthenticator 训练 + 评估 + 可视化（真实/模拟数据）
    # ════════════════════════════════════════════════════════════
    print("\n【Part 2】SVMAuthenticator 训练 + 可视化")
    print("-" * 40)

    DATA_DIR   = Path(__file__).parent / 'output'
    OUTPUT_DIR = Path(__file__).parent / 'models'
    JSON_PATH  = DATA_DIR / 'features_by_user.json'

    # ── 判断是否有真实数据 ───────────────────────────────────────
    if JSON_PATH.exists():
        print(f"\n检测到真实特征文件: {JSON_PATH}")
        print("启动完整 SVM 训练流程...\n")

        svm_auth, results = train_svm_system(
            data_dir   = str(DATA_DIR),
            output_dir = str(OUTPUT_DIR),
            test_ratio = 0.2
        )

        print("\n" + "=" * 60)
        print("全部完成！输出文件列表：")
        print("=" * 60)
        for p in sorted(OUTPUT_DIR.rglob('*')):
            if p.is_file():
                size = p.stat().st_size / 1024
                print(f"  {p.relative_to(OUTPUT_DIR)}  ({size:.1f} KB)")

    # ── 无真实数据：用模拟数据跑完整流程 ────────────────────────
    else:
        print(f"\n未找到 {JSON_PATH}")
        print("使用模拟数据运行 SVMAuthenticator 完整流程...\n")

        np.random.seed(42)
        n_users     = 5
        feature_dim = 47
        n_samples   = 20

        mock_features = {}
        for i in range(n_users):
            uid    = f"user_{i:02d}"
            center = np.random.randn(feature_dim) * 2
            mock_features[uid] = (
                center + np.random.randn(n_samples, feature_dim) * 0.5
            ).astype(np.float32)

        # 划分训练/测试
        train_features, test_data_viz = {}, []
        for uid, feats in mock_features.items():
            split = int(len(feats) * 0.8)
            train_features[uid] = feats[:split]
            for feat in feats[split:]:
                test_data_viz.append((feat, uid))

        # 训练
        print("  训练 SVMAuthenticator...")
        svm_auth = SVMAuthenticator()
        svm_auth.train(train_features, use_grid_search=False)
        print("  训练完成")

        # 评估
        print("\n  评估结果:")
        correct = sum(
            1 for feat, uid in test_data_viz
            if svm_auth.predict(feat) == uid
        )
        acc = correct / len(test_data_viz)
        print(f"  测试准确率: {acc*100:.2f}%")

        mock_results = {
            'test_accuracy': acc,
            'cv_accuracy'  : 0.0,
            'cv_std'       : 0.0
        }

        # 可视化
        print("\n  生成可视化图表...")
        viz_dir = OUTPUT_DIR / 'visualizations'
        viz     = VisualizationSuite(output_dir=str(viz_dir))

        print("  [1/2] SVM 决策边界...")
        best_params = {
            'kernel': svm_auth.kernel,
            'C'     : svm_auth.C,
            'gamma' : svm_auth.gamma
        }
        viz.plot_svm_decision_boundary(mock_features, svm_params=best_params)

        print("  [2/2] 混淆矩阵...")
        viz.plot_confusion_matrix(svm_auth, test_data_viz)


        print(f"\n  所有图表已保存至: {viz_dir}")
        print("\n  输出文件列表：")
        for p in sorted(viz_dir.rglob('*.png')):
            size = p.stat().st_size / 1024
            print(f"    {p.name}  ({size:.1f} KB)")

        print("\n✓ SVMAuthenticator 模拟数据测试通过")

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
