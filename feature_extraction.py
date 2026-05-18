# feature_extraction.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.signal import freqz, find_peaks
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not found, using simplified MFCC implementation")


class FeatureExtractor:

    # 子带定义 (与Event Detection一致)
    SUB_BANDS = [
        (150, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 3000)
    ]

    def __init__(self, sr=44100, n_mfcc=13, frame_size=0.025, hop_size=0.01):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.frame_len = int(frame_size * sr)
        self.hop_len = int(hop_size * sr)

    def bandpass_filter(self, signal, low_freq, high_freq, order=4):
        nyq = self.sr / 2
        low = max(low_freq / nyq, 0.001)
        high = min(high_freq / nyq, 0.999)

        if low >= high:
            return np.zeros_like(signal)

        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered

    def compute_subband_energy(self, signal):
        energies = np.zeros(len(self.SUB_BANDS))

        for i, (low, high) in enumerate(self.SUB_BANDS):
            filtered = self.bandpass_filter(signal, low, high)
            energies[i] = np.sum(filtered ** 2)

        return energies

    def extract_ser(self, signal):
        energies = self.compute_subband_energy(signal)
        total_energy = np.sum(energies) + 1e-10  # 防止除零

        ser = energies / total_energy
        return ser

    def extract_pcc(self, left_signal, right_signal):
        pcc = np.zeros(len(self.SUB_BANDS))

        for i, (low, high) in enumerate(self.SUB_BANDS):
            # 子带滤波
            left_filtered = self.bandpass_filter(left_signal, low, high)
            right_filtered = self.bandpass_filter(right_signal, low, high)

            # 计算Pearson相关系数
            if np.std(left_filtered) > 1e-10 and np.std(right_filtered) > 1e-10:
                correlation = np.corrcoef(left_filtered, right_filtered)[0, 1]
                pcc[i] = correlation if not np.isnan(correlation) else 0
            else:
                pcc[i] = 0

        return pcc

    def extract_mfcc(self, signal):
        if HAS_LIBROSA:
            # 使用librosa计算MFCC
            mfcc = librosa.feature.mfcc(y=signal, sr=self.sr,
                                        n_mfcc=self.n_mfcc,
                                        n_fft=self.frame_len,
                                        hop_length=self.hop_len)

            # 计算Delta和Delta-Delta
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # 拼接 (n_mfcc*3, n_frames) -> (n_frames, n_mfcc*3)
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
        else:
            # 简化实现：使用基本FFT
            mfcc_features = self._simple_mfcc(signal)

        # 计算帧平均
        mfcc_mean = np.mean(mfcc_features, axis=0)

        expected_dim = self.n_mfcc * 3
        if len(mfcc_mean) != expected_dim:
            mfcc_mean = np.zeros(expected_dim)

        return mfcc_features, mfcc_mean

    def _simple_mfcc(self, signal):
        """
        简化的MFCC实现（当librosa不可用时）

        使用FFT + Mel滤波器组 + DCT的基本流程
        """
        n_frames = (len(signal) - self.frame_len) // self.hop_len + 1
        n_fft = self.frame_len

        # Mel滤波器组参数
        n_mels = 26
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sr / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)

        # 创建Mel滤波器组
        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            for j in range(left, center):
                if center > left:
                    fbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    fbank[i - 1, j] = (right - j) / (right - center)

        # 分帧处理
        mfcc_all = []
        for i in range(n_frames):
            start = i * self.hop_len
            frame = signal[start:start + self.frame_len]

            # 加窗
            frame = frame * np.hamming(len(frame))

            # FFT
            spectrum = np.abs(np.fft.rfft(frame, n_fft)) ** 2

            # Mel滤波
            mel_spectrum = np.dot(fbank, spectrum)
            mel_spectrum = np.log(mel_spectrum + 1e-10)

            # DCT得到MFCC
            mfcc = np.zeros(self.n_mfcc)
            for j in range(self.n_mfcc):
                mfcc[j] = np.sum(mel_spectrum * np.cos(
                    np.pi * j * (np.arange(n_mels) + 0.5) / n_mels))

            mfcc_all.append(mfcc)

        mfcc_all = np.array(mfcc_all)

        # 计算Delta
        delta = np.zeros_like(mfcc_all)
        for i in range(2, len(mfcc_all) - 2):
            delta[i] = (mfcc_all[i + 1] - mfcc_all[i - 1] +
                       2 * (mfcc_all[i + 2] - mfcc_all[i - 2])) / 10

        # 计算Delta-Delta
        delta2 = np.zeros_like(delta)
        for i in range(2, len(delta) - 2):
            delta2[i] = (delta[i + 1] - delta[i - 1] +
                        2 * (delta[i + 2] - delta[i - 2])) / 10

        # 拼接
        mfcc_features = np.hstack([mfcc_all, delta, delta2])

        return mfcc_features

    def extract_bone_conduction_descriptor(self, left_signal, right_signal=None):
        # SER (始终计算)
        ser = self.extract_ser(left_signal)

        if right_signal is not None:
            # 双通道：计算PCC
            pcc = self.extract_pcc(left_signal, right_signal)
            # 同时计算右通道SER并平均
            ser_right = self.extract_ser(right_signal)
            ser = (ser + ser_right) / 2
            bcd = np.concatenate([ser, pcc])
        else:
            bcd = ser

        return bcd

    def extract_respiratory_tract_descriptor(self, left_signal, right_signal=None):
        _, mfcc_left = self.extract_mfcc(left_signal)

        if right_signal is not None:
            _, mfcc_right = self.extract_mfcc(right_signal)
            rtd = np.concatenate([mfcc_left, mfcc_right])
        else:
            rtd = mfcc_left

        return rtd

    def extract_features(self, left_signal, right_signal=None, sr=None):
        if sr is not None:
            self.sr = sr
            self.frame_len = int(self.frame_size * sr)
            self.hop_len = int(self.hop_size * sr)

        # Level I: BCD (8维双通道 / 4维单通道)
        bcd = self.extract_bone_conduction_descriptor(left_signal, right_signal)

        # Level II: RTD — 只用均值，不加std，不拼右通道
        rtd = self.extract_respiratory_tract_descriptor(left_signal)
        # 单通道MFCC均值 = 39维，保持简洁

        features = np.concatenate([bcd, rtd])

        feature_info = {
            'bcd_dim': len(bcd),
            'rtd_dim': len(rtd),
            'spec_dim': 0,
            'total_dim': len(features),
            'is_stereo': right_signal is not None
        }
        return features, feature_info

    def _extract_rtd_enhanced(self, left_signal, right_signal=None):
        """
        增强版 RTD：mean + std 各 39 维 = 78 维/通道
        """
        mfcc_features_left, _ = self.extract_mfcc(left_signal)
        mfcc_left = np.concatenate([
            np.mean(mfcc_features_left, axis=0),  # (39,)
            np.std(mfcc_features_left, axis=0),  # (39,)
        ])  # (78,)

        if right_signal is not None:
            mfcc_features_right, _ = self.extract_mfcc(right_signal)
            mfcc_right = np.concatenate([
                np.mean(mfcc_features_right, axis=0),
                np.std(mfcc_features_right, axis=0),
            ])  # (78,)
            return np.concatenate([mfcc_left, mfcc_right])  # (156,)
        else:
            return mfcc_left  # (78,)

    def extract_spectral_features(self, signal):
        """
        频谱统计特征：centroid / bandwidth / rolloff 各 mean+std = 6 维
        """
        if not HAS_LIBROSA:
            return np.zeros(6)

        centroid = librosa.feature.spectral_centroid(
            y=signal, sr=self.sr,
            n_fft=self.frame_len, hop_length=self.hop_len)[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=signal, sr=self.sr,
            n_fft=self.frame_len, hop_length=self.hop_len)[0]
        rolloff = librosa.feature.spectral_rolloff(
            y=signal, sr=self.sr,
            n_fft=self.frame_len, hop_length=self.hop_len)[0]

        return np.array([
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
        ])  # (6,)

    def extract_from_breath_cycle(self, breath_cycle, left_signal, right_signal=None):
        start_sample = int(breath_cycle['start_time'] * self.sr)
        end_sample = int(breath_cycle['end_time'] * self.sr)
        end_sample = min(end_sample, len(left_signal))

        left_segment = left_signal[start_sample:end_sample]

        right_segment = None
        if right_signal is not None:
            right_segment = right_signal[start_sample:end_sample]

        return self.extract_features(left_segment, right_segment)

    def extract_from_all_cycles(self, breath_cycles, left_signal, right_signal=None):
        all_features = []

        for cycle in breath_cycles:
            features, _ = self.extract_from_breath_cycle(
                cycle, left_signal, right_signal)
            all_features.append(features)

        if len(all_features) == 0:
            return np.array([]), np.array([])

        all_features = np.array(all_features)
        mean_features = np.mean(all_features, axis=0)

        return all_features, mean_features

    def save_features(self, features_by_user, path='features_by_user.json'):
        import json
        features_save = {uid: feat.tolist() for uid, feat in features_by_user.items()}
        with open(path, 'w') as f:
            json.dump(features_save, f)
        print(f"特征已保存: {path}, 共 {len(features_by_user)} 个用户")

    def load_features(self, path='features_by_user.json'):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        features_by_user = {uid: np.array(feat) for uid, feat in data.items()}
        print(f"特征已加载: {path}, 共 {len(features_by_user)} 个用户")
        return features_by_user


def plot_subject_envelopes_left_zoom(output_path='output'):
    output_dir = Path(output_path)
    subjects = [d for d in output_dir.iterdir() if d.is_dir()]

    if not subjects:
        print("未发现测试者文件夹。")
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.15)

    extractor = FeatureExtractor()

    # ── 高明度、区分度大的颜色列表 ──────────────────────────────
    DISTINCT_COLORS = [
        '#E63946',  # 红
        '#F4A261',  # 橙
        '#2A9D8F',  # 青绿
        '#457B9D',  # 钢蓝
        '#8338EC',  # 紫
        '#FB5607',  # 深橙
        '#06D6A0',  # 薄荷绿
        '#118AB2',  # 蓝
        '#FFD166',  # 黄
        '#EF476F',  # 玫红
        '#56CFE1',  # 天蓝
        '#80B918',  # 黄绿
        '#9B5DE5',  # 淡紫
        '#F15BB5',  # 粉
        '#00BBF9',  # 亮蓝
    ]

    # ── 匿名映射 ────────────────────────────────────────────────
    subjects = sorted(subjects)                                        # ← 排序保证顺序稳定
    anon_names = [f"user_{i}" for i in range(len(subjects))]
    name_map   = {d.name: anon for d, anon in zip(subjects, anon_names)}

    axins = inset_axes(ax, width="38%", height="55%", loc='center left',
                       bbox_to_anchor=(-0.48, 0, 1, 1),
                       bbox_transform=ax.transAxes)

    for i, subject_dir in enumerate(subjects):
        wav_files = list(subject_dir.glob('*.wav')) + list(subject_dir.glob('*.WAV'))
        if not wav_files:
            continue

        subject_all_mfccs = []
        for wav_path in wav_files:
            sr, data = wavfile.read(str(wav_path))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            signal = data[:, 0] if len(data.shape) == 2 else data
            mfcc_feat, _ = extractor.extract_mfcc(signal)
            subject_all_mfccs.append(np.mean(mfcc_feat[:, :13], axis=0))

        subject_mean_envelope = np.mean(subject_all_mfccs, axis=0)

        current_color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]  # ← 换色表
        display_name  = name_map[subject_dir.name]                  # ← 匿名

        ax.plot(range(1, 14), subject_mean_envelope, marker='o',
                label=display_name,                                 # ← 匿名标签
                color=current_color, linewidth=2)

        axins.plot(range(1, 14), subject_mean_envelope, marker='o',
                   color=current_color, linewidth=2.5)

    axins.set_xlim(1.8, 4.2)
    axins.set_ylim(-50, 260)
    axins.set_title("Zoomed Discriminative Region (2-4)", fontsize=15, pad=10)
    axins.set_ylabel('Magnitude (Log Energy)', fontsize=15)
    axins.grid(True, linestyle=':', alpha=0.5)
    axins.tick_params(axis='both', which='major', labelsize=9)

    ax.set_title('Acoustic Envelopes Comparison', fontsize=17, pad=20)
    ax.set_xlabel('MFCC Coefficients (1-13)', fontsize=16)
    ax.set_ylabel('Magnitude (Log Energy)', fontsize=16)
    ax.set_xticks(range(1, 14))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=16, frameon=True, shadow=True, ncol=2)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",
               alpha=0.6, linestyle='--', lw=3.0)

    save_path = 'subject_envelope_left_zoom_fixed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    print(f"图像修复完成并保存至: {save_path}")



def plot_bcd_bar_comparison_refined(output_path='output'):
    output_dir = Path(output_path)
    selected_subjects = ['DingYuxin', 'ChenSiming', 'JiangShenyao', 'GuoKaiwen', 'GaoShang']

    # ── 匿名映射 ────────────────────────────────────────────────
    anon_names = [f"user_{i}" for i in range(len(selected_subjects))]
    name_map   = {real: anon for real, anon in zip(selected_subjects, anon_names)}

    extractor = FeatureExtractor()
    labels = ['150-500Hz', '500-1000Hz', '1000-1500Hz', '1500-3000Hz']

    means = []
    stds  = []

    print("正在从真实数据提取子带能量比 (SER)...")
    for name in selected_subjects:
        sub_dir   = output_dir / name
        wav_files = list(sub_dir.glob('*.wav')) + list(sub_dir.glob('*.WAV'))

        if not wav_files:
            print(f"Warning: No WAV files found for {name_map[name]}")  # ← 匿名
            continue

        current_ser = []
        for wav_path in wav_files:
            sr, data = wavfile.read(str(wav_path))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            signal = data[:, 0] if len(data.shape) == 2 else data
            current_ser.append(extractor.extract_ser(signal))

        means.append(np.mean(current_ser, axis=0))
        stds.append(np.std(current_ser, axis=0))

    x            = np.arange(len(labels))
    num_subjects = len(selected_subjects)
    width        = 0.8 / num_subjects

    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ['#2E86AB', '#A8C256', '#F6AE2D', '#57B8A4', '#8BC4E0']

    for i, name in enumerate(selected_subjects):
        offset = (i - (num_subjects - 1) / 2) * width
        bars = ax.bar(
            x + offset, means[i], width, yerr=stds[i],
            label=name_map[name],              # ← 匿名标签
            color=colors[i], capsize=3, edgecolor='black', alpha=0.8
        )

        for bar, mean_val in zip(bars, means[i]):
            bar_height = bar.get_height()
            bar_x      = bar.get_x() + bar.get_width() / 2

            if bar_height > 0.15:
                ax.text(
                    bar_x, bar_height - 0.03,
                    f'{mean_val:.2f}',
                    ha='center', va='top',
                    fontsize=7.5, color='white', fontweight='bold',
                )
            else:
                yerr_val = stds[i][list(bars).index(bar)]
                ax.text(
                    bar_x, bar_height + yerr_val + 0.015,
                    f'{mean_val:.2f}',
                    ha='center', va='bottom',
                    fontsize=7.5, color='black',
                )

    ax.set_title('Level 1: Sub-band Energy Ratio (SER) - Individual Comparison',
                 fontsize=14, pad=15)
    ax.set_ylabel('Energy Ratio (Normalized)', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('level1_bar_refined.png', dpi=300)
    plt.show()
    print("✓ Level 1 多人对比条形图已生成。")


def plot_pcc_heatmap_analysis(output_path='output'):
    output_dir = Path(output_path)
    # 选取受试者实际姓名
    selected_subjects = ['DingYuxin', 'ChenSiming', 'JiangShenyao', 'GuoKaiwen', 'GaoShang']

    # 创建一个用户标识符的映射
    user_labels = ['user_0', 'user_1', 'user_2', 'user_3', 'user_4']

    extractor = FeatureExtractor()
    labels = ['150-500Hz', '500-1000Hz', '1000-1500Hz', '1500-3000Hz']

    pcc_matrix = []

    print("正在提取双通道 PCC 特征...")
    for name in selected_subjects:
        sub_dir = output_dir / name
        wav_files = list(sub_dir.glob('*.wav'))

        current_pcc_list = []
        for wav_path in wav_files:
            sr, data = wavfile.read(str(wav_path))
            if len(data.shape) < 2:
                continue  # PCC 需要双通道数据

            data = data.astype(np.float32) / 32768.0
            left, right = data[:, 0], data[:, 1]

            # 调用你代码里的 extract_pcc
            pcc = extractor.extract_pcc(left, right)
            current_pcc_list.append(pcc)

        if current_pcc_list:
            pcc_matrix.append(np.mean(current_pcc_list, axis=0))
        else:
            pcc_matrix.append(np.zeros(4))  # 防止缺失数据

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.heatmap(pcc_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=user_labels)  # 用 user_labels 替换 yticklabels

    plt.title('Level 1: Pearson Correlation Coefficient (PCC) Heatmap', fontsize=14, pad=15)
    plt.xlabel('Sub-bands (Hz)', fontsize=15)
    plt.ylabel('Subjects', fontsize=15)

    plt.tight_layout()
    plt.savefig('level1_pcc_heatmap.png', dpi=300)
    plt.show()
    print("✓ PCC 热力图已生成。")


def plot_respiratory_formant_comparison(project_root, n_coeffs=16):
    root_path = Path(project_root)
    output_root = root_path / "output"
    figures_dir = root_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 10))

    # ── 高明度、区分度大的颜色列表 ──────────────────────────────
    DISTINCT_COLORS = [
        '#E63946',  # 红
        '#F4A261',  # 橙
        '#2A9D8F',  # 青绿
        '#457B9D',  # 钢蓝
        '#8338EC',  # 紫
        '#FB5607',  # 深橙
        '#06D6A0',  # 薄荷绿
        '#118AB2',  # 蓝
        '#FFD166',  # 黄
        '#EF476F',  # 玫红
        '#56CFE1',  # 天蓝
        '#80B918',  # 黄绿
        '#9B5DE5',  # 淡紫
        '#F15BB5',  # 粉
        '#00BBF9',  # 亮蓝
    ]

    subjects = sorted([d.name for d in output_root.iterdir() if d.is_dir()])
    if not subjects:
        print("Error: No subject folders found in 'output'.")
        return

    # ── 匿名映射 ────────────────────────────────────────────────
    anon_names = [f"user_{i}" for i in range(len(subjects))]
    name_map   = {real: anon for real, anon in zip(subjects, anon_names)}

    def estimate_lpc(signal, order):
        r = np.correlate(signal, signal, mode='full')[len(signal) - 1:]
        r = r[:order + 1]
        from scipy.linalg import toeplitz, solve
        phi = toeplitz(r[:-1])
        return solve(phi, r[1:])

    for i, sub_name in enumerate(subjects):
        sub_dir  = output_root / sub_name
        wav_files = list(sub_dir.glob('*.wav'))
        if not wav_files:
            continue

        fs, data = wavfile.read(str(wav_files[0]))
        if len(data.shape) > 1:
            data = data[:, 0]
        data  = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-6)
        d_pre = np.append(data[0], data[1:] - 0.97 * data[:-1])

        lpc_coeffs   = estimate_lpc(d_pre, n_coeffs)
        a            = np.concatenate(([1], -lpc_coeffs))
        w, h         = freqz(1, a, worN=8000)
        freqs        = w * fs / (2 * np.pi)
        log_envelope = 20 * np.log10(np.abs(h) + 1e-6)
        log_envelope -= np.max(log_envelope)

        current_color = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]  # ← 换色表
        display_name  = name_map[sub_name]                          # ← 匿名

        plt.plot(freqs, log_envelope,
                 label=display_name,          # ← 匿名标签
                 color=current_color, linewidth=2.0)

        peaks, _ = find_peaks(log_envelope, distance=len(freqs) // 10)
        if len(peaks) > 0:
            f1_idx  = peaks[0]
            f1_freq = freqs[f1_idx]
            plt.scatter(f1_freq, log_envelope[f1_idx],
                        color=current_color, zorder=5, s=40)

            if i % 2 == 0:
                y_offset = 25 + (i // 2 * 20)
                va = 'bottom'
            else:
                y_offset = -35 - (i // 2 * 20)
                va = 'top'

            plt.annotate(f'{int(f1_freq)}Hz',
                         (f1_freq, log_envelope[f1_idx]),
                         textcoords="offset points",
                         xytext=(0, y_offset),
                         ha='center', va=va,
                         fontsize=12, fontweight='bold',
                         color=current_color,
                         arrowprops=dict(
                             arrowstyle='->',
                             color=current_color,
                             lw=1, alpha=0.6,
                             shrinkA=0, shrinkB=2,
                         ))

    plt.title('Level 2 Analysis: Comparative Respiratory Formants',
              fontsize=20, pad=40)
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Normalized Magnitude (dB)', fontsize=16)
    plt.xlim(0, 3500)
    plt.ylim(-65, 30)
    plt.legend(loc='lower right', fontsize=13,
               frameon=True, shadow=True, ncol=3)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    save_file = figures_dir / "comparative_formants_all_subjects.png"
    plt.savefig(str(save_file), dpi=300)
    plt.show()
    print(f"Success: Plot saved to {save_file}")




if __name__ == '__main__':
    # 测试代码
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from pathlib import Path
    from scipy.io import wavfile

    # 找测试文件
    output_dir = Path(__file__).parent / 'output'
    subjects = [d for d in output_dir.iterdir() if d.is_dir()]

    if subjects:
        test_subject = subjects[0]
        wav_files = list(test_subject.glob('*.wav')) + list(test_subject.glob('*.WAV'))

        if wav_files:
            test_file = wav_files[0]
            print(f'测试文件: {test_file}')

            # 加载音频
            sr, data = wavfile.read(str(test_file))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0

            if len(data.shape) == 2:
                left = data[:, 0]
                right = data[:, 1]
            else:
                left = data
                right = None

            print(f'采样率: {sr} Hz')
            print(f'双通道: {right is not None}')

            # 特征提取
            extractor = FeatureExtractor(sr=sr)
            features, info = extractor.extract_features(left, right)

            print(f'\n特征提取结果:')
            print(f'  BCD维度: {info["bcd_dim"]}')
            print(f'  RTD维度: {info["rtd_dim"]}')
            print(f'  总维度: {info["total_dim"]}')
            print(f'  特征范围: [{features.min():.4f}, {features.max():.4f}]')

            # 测试SER
            ser = extractor.extract_ser(left)
            print(f'\nSER特征: {ser}')
            print(f'  SER和: {ser.sum():.4f} (应为1.0)')

            print('\n✓ 特征提取模块测试通过')

    print("\n正在生成全局分析图表...")
    plot_subject_envelopes_left_zoom()
    plot_bcd_bar_comparison_refined()
    plot_pcc_heatmap_analysis()


    project_root_path = "/Users/yaojinru/Desktop/breathsign_denoise 3"
    plot_respiratory_formant_comparison(project_root_path)

# ── 提取所有用户特征并保存 ──────────────────────────
    extractor = FeatureExtractor(sr=sr)
    features_by_user = {}

    for subject_dir in sorted(subjects):
        uid = subject_dir.name
        wav_files = list(subject_dir.glob('*.wav')) + list(subject_dir.glob('*.WAV'))
        user_feats = []

        for wav_path in wav_files:
            sr_w, data = wavfile.read(str(wav_path))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            left  = data[:, 0] if len(data.shape) == 2 else data
            right = data[:, 1] if len(data.shape) == 2 else None

            feat, _ = extractor.extract_features(left, right, sr=sr_w)
            user_feats.append(feat)

        dims = [f.shape[0] for f in user_feats]
        print(f"  {uid}: 维度分布 {set(dims)}")  # 应该只有一个数字

        if user_feats:
            ref_dim = max(set(dims), key=dims.count)  # 取出现最多的维度作为基准
            valid_feats = [f for f in user_feats if f.shape[0] == ref_dim]

            if len(valid_feats) != len(user_feats):
                print(f"  {uid}: 丢弃 {len(user_feats) - len(valid_feats)} 个维度异常样本 (保留维度={ref_dim})")

            features_by_user[uid] = np.array(valid_feats)
            print(f"  {uid}: {len(valid_feats)} 样本, 维度 {valid_feats[0].shape}")

    # 保存到 JSON
    save_path = str(output_dir / 'features_by_user.json')
    extractor.save_features(features_by_user, save_path)
    print(f"\n✓ 特征已保存至: {save_path}")
