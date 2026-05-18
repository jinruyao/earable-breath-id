# event_detection.py

from scipy.signal import stft
from sklearn.mixture import GaussianMixture
import warnings
import matplotlib.pyplot as plt
import scipy.stats as stats
warnings.filterwarnings('ignore')


class BreathEventDetector:
    """
    BreathSign 呼吸事件检测器

    基于GMM的自适应分割算法，从降噪后的音频信号中检测呼吸事件，
    并识别吸气/呼气相位。

    Attributes:
        sr (int): 采样率(Hz)
        frame_size (float): 帧长(秒)
        hop_size (float): 帧移(秒)
        sub_bands (list): 子带频率范围列表
        sub_band_weights (list): 子带权重，低频子带权重更高
        min_breath_duration (float): 最小呼吸相位时长(秒)
        gmm_components (int): GMM组件数

    References:
        [1] BreathSign, IEEE INFOCOM 2023, Section III-C
    """

    # 子带定义 (论文 Section III-C)
    SUB_BANDS = [
        (150, 500),    # 子带1: 低频，包含主要呼吸能量
        (500, 1000),   # 子带2: 中低频
        (1000, 1500),  # 子带3: 中高频
        (1500, 3000)   # 子带4: 高频
    ]

    # 子带权重 (论文 Section III-C)
    # 低频子带权重更高，因为呼吸声能量主要集中在低频
    SUB_BAND_WEIGHTS = [0.7, 0.2, 0.08, 0.02]

    def __init__(self, sr=44100, frame_size=0.025, hop_size=0.01,
                 min_breath_duration=1.0, gmm_components=2):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_breath_duration = min_breath_duration
        self.gmm_components = gmm_components

        # 计算帧参数
        self.frame_len = int(frame_size * sr)
        self.hop_len = int(hop_size * sr)

    def compute_subband_energy(self, signal):
        # STFT参数
        nperseg = self.frame_len
        noverlap = self.frame_len - self.hop_len

        # 计算STFT
        frequencies, times, Zxx = stft(signal, fs=self.sr,
                                       nperseg=nperseg, noverlap=noverlap)

        # 计算功率谱
        power_spectrum = np.abs(Zxx) ** 2

        # 计算各子带能量
        n_frames = power_spectrum.shape[1]
        subband_energies = np.zeros((n_frames, len(self.SUB_BANDS)))

        for i, (low, high) in enumerate(self.SUB_BANDS):
            # 找到对应频率范围的索引
            freq_mask = (frequencies >= low) & (frequencies < high)
            # 计算该子带的能量
            band_energy = np.sum(power_spectrum[freq_mask, :], axis=0)
            # 对数能量 (加小值防止log(0))
            subband_energies[:, i] = np.log(band_energy + 1e-10)

        return subband_energies, times

    def gmm_segmentation(self, subband_energies):
        n_frames = subband_energies.shape[0]
        subband_probs = np.zeros((n_frames, len(self.SUB_BANDS)))

        self.gmm_params_record = []

        for i in range(len(self.SUB_BANDS)):
            # 获取该子带的能量特征
            features = subband_energies[:, i].reshape(-1, 1)

            # 训练GMM
            gmm = GaussianMixture(n_components=self.gmm_components,
                                  covariance_type='full',
                                  random_state=42,
                                  max_iter=100)
            gmm.fit(features)

            # 计算后验概率
            probs = gmm.predict_proba(features)

            # 确定哪个组件是呼吸类 (能量较高的组件)
            if gmm.means_[0] > gmm.means_[1]:
                breath_component = 0
                noise_component = 1
            else:
                breath_component = 1
                noise_component = 0

            self.gmm_params_record.append({
                'mu_0': gmm.means_[noise_component][0],
                'mu_1': gmm.means_[breath_component][0],
                'sigma_0': gmm.covariances_[noise_component][0][0],
                'sigma_1': gmm.covariances_[breath_component][0][0]
            })

            # 保存该子带的呼吸后验概率
            subband_probs[:, i] = probs[:, breath_component]

        # 加权融合 - 使用论文的子带权重 βᵢ
        weights = np.array(self.SUB_BAND_WEIGHTS)

        # 加权融合概率
        breath_prob = np.dot(subband_probs, weights)

        # 使用概率阈值进行决策（与可视化一致）
        # 阈值0.4：比0.5更宽松，避免漏检高概率段
        prob_threshold = 0.4
        is_breathing = breath_prob > prob_threshold

        # 平滑处理：使用滑动窗口消除孤立点
        window_size = 5
        smoothed = np.convolve(is_breathing.astype(float),
                              np.ones(window_size)/window_size, mode='same')
        is_breathing = smoothed > 0.5

        return breath_prob, is_breathing

    def duration_check(self, is_breathing, times, merge_gap=0.3):
        filtered_breathing = is_breathing.copy()
        raw_segments = []

        # Step 1: 找出所有连续段
        in_segment = False
        start_idx = 0

        for i in range(len(is_breathing)):
            if is_breathing[i] and not in_segment:
                in_segment = True
                start_idx = i
            elif not is_breathing[i] and in_segment:
                in_segment = False
                raw_segments.append((start_idx, i))

        if in_segment:
            raw_segments.append((start_idx, len(is_breathing)))

        # Step 2: 合并间隔很短的相邻段
        if len(raw_segments) <= 1:
            merged_segments = raw_segments
        else:
            merged_segments = [raw_segments[0]]
            for start_idx, end_idx in raw_segments[1:]:
                prev_start, prev_end = merged_segments[-1]
                gap = times[start_idx] - times[prev_end - 1] if prev_end < len(times) else 0

                if gap < merge_gap:
                    # 间隔小于阈值，合并
                    merged_segments[-1] = (prev_start, end_idx)
                    # 填补间隙
                    filtered_breathing[prev_end:start_idx] = True
                else:
                    merged_segments.append((start_idx, end_idx))

        # Step 3: 时长检查
        segments = []
        for start_idx, end_idx in merged_segments:
            duration = times[min(end_idx - 1, len(times) - 1)] - times[start_idx]

            if duration >= self.min_breath_duration:
                segments.append((start_idx, end_idx, duration))
            else:
                filtered_breathing[start_idx:end_idx] = False

        return filtered_breathing, segments

    def identify_breath_phase(self, signal, segments, times):
        """
        呼吸相位识别 - 段内检测吸气/呼气转换

        对每个检测到的呼吸段，在段内检测吸气→呼气的转换点。
        每个足够长的呼吸段被视为包含完整的吸气+呼气周期。

        算法原理:
            - 吸气通常能量较低（被动过程）
            - 呼气通常能量较高（主动过程）
            - 在段内找能量变化最大的点作为转换点

        Args:
            signal: 原始信号
            segments: 呼吸段列表
            times: 各帧时间点

        Returns:
            breath_events: 呼吸事件列表（每个段产生2个事件：吸气+呼气）
        """
        if len(segments) == 0:
            return []

        breath_events = []

        for start_idx, end_idx, duration in segments:
            start_sample = int(times[start_idx] * self.sr)
            end_sample = int(times[min(end_idx - 1, len(times) - 1)] * self.sr)
            end_sample = min(end_sample, len(signal))
            start_sample = min(start_sample, end_sample)

            if end_sample <= start_sample:
                continue

            segment_signal = signal[start_sample:end_sample]
            segment_len = len(segment_signal)

            # 在段内检测吸气/呼气转换点
            # 将段分成若干窗口，计算能量，找能量上升最明显的点
            n_windows = max(4, segment_len // (self.sr // 20))  # 约50ms窗口
            window_size = segment_len // n_windows

            window_energies = []
            for w in range(n_windows):
                w_start = w * window_size
                w_end = min((w + 1) * window_size, segment_len)
                if w_end > w_start:
                    window_energies.append(np.mean(segment_signal[w_start:w_end] ** 2))

            # 找能量变化的转换点
            if len(window_energies) >= 2:
                energy_diff = np.diff(window_energies)
                # 找能量上升最明显的点（吸气→呼气转换）
                transition_window = np.argmax(np.abs(energy_diff)) + 1
                transition_sample = start_sample + transition_window * window_size
            else:
                # 默认中点
                transition_sample = (start_sample + end_sample) // 2

            # 确保转换点在合理范围内（不在边缘）
            min_phase_samples = int(0.2 * self.sr)  # 最小200ms
            transition_sample = max(start_sample + min_phase_samples,
                                   min(transition_sample, end_sample - min_phase_samples))

            transition_time = transition_sample / self.sr
            start_time = times[start_idx]
            end_time = times[min(end_idx - 1, len(times) - 1)]

            # 计算转换点对应的帧索引
            transition_idx = start_idx + int((transition_sample - start_sample) / self.sr / self.hop_size)
            transition_idx = min(max(transition_idx, start_idx + 1), end_idx - 1)

            # 吸气事件（前半部分）
            inhalation_signal = segment_signal[:transition_sample - start_sample]
            breath_events.append({
                'start_time': start_time,
                'end_time': transition_time,
                'start_idx': start_idx,
                'end_idx': transition_idx,
                'phase': 'inhalation',
                'energy': np.mean(inhalation_signal ** 2) if len(inhalation_signal) > 0 else 0
            })

            # 呼气事件（后半部分）
            exhalation_signal = segment_signal[transition_sample - start_sample:]
            breath_events.append({
                'start_time': transition_time,
                'end_time': end_time,
                'start_idx': transition_idx,
                'end_idx': end_idx,
                'phase': 'exhalation',
                'energy': np.mean(exhalation_signal ** 2) if len(exhalation_signal) > 0 else 0
            })

        return breath_events

    def extract_breath_cycles(self, signal, breath_events):
        """
        提取完整呼吸周期

        将检测到的呼吸事件组合成完整的呼吸周期（吸气+呼气）。

        Args:
            signal: 原始信号
            breath_events: 呼吸事件列表

        Returns:
            breath_cycles: 呼吸周期列表
                [{
                    'inhalation': {...},
                    'exhalation': {...},
                    'signal': numpy array,
                    'start_time': float,
                    'end_time': float
                }, ...]
        """
        breath_cycles = []

        i = 0
        while i < len(breath_events) - 1:
            current = breath_events[i]
            next_event = breath_events[i + 1]

            # 寻找吸气-呼气配对
            if current['phase'] == 'inhalation' and next_event['phase'] == 'exhalation':
                # 提取信号片段
                start_sample = int(current['start_time'] * self.sr)
                end_sample = int(next_event['end_time'] * self.sr)
                end_sample = min(end_sample, len(signal))

                cycle_signal = signal[start_sample:end_sample]

                breath_cycles.append({
                    'inhalation': current,
                    'exhalation': next_event,
                    'signal': cycle_signal,
                    'start_time': current['start_time'],
                    'end_time': next_event['end_time']
                })
                i += 2
            else:
                i += 1

        return breath_cycles

    def detect(self, signal, sr=None):
        """
        完整的呼吸事件检测流程

        按顺序执行三阶段检测:
            Step 1: GMM Segmentation → Step 2: Duration Check → Step 3: Phase Identification

        Args:
            signal: 降噪后的音频信号
            sr: 采样率(Hz)，None则使用初始化时的值

        Returns:
            result: 检测结果字典
                {
                    'breath_events': 呼吸事件列表,
                    'breath_cycles': 呼吸周期列表,
                    'breath_prob': 各帧呼吸概率,
                    'is_breathing': 二值分割结果,
                    'times': 各帧时间点
                }
        """
        if sr is not None:
            self.sr = sr
            self.frame_len = int(self.frame_size * sr)
            self.hop_len = int(self.hop_size * sr)

        # Step 1: 计算子带能量并进行GMM分割
        subband_energies, times = self.compute_subband_energy(signal)
        breath_prob, is_breathing = self.gmm_segmentation(subband_energies)

        # Step 2: 时长检查
        filtered_breathing, segments = self.duration_check(is_breathing, times)

        # Step 3: 相位识别
        breath_events = self.identify_breath_phase(signal, segments, times)

        # 提取完整呼吸周期
        breath_cycles = self.extract_breath_cycles(signal, breath_events)

        return {
            'breath_events': breath_events,
            'breath_cycles': breath_cycles,
            'breath_prob': breath_prob,
            'is_breathing': filtered_breathing,
            'times': times,
            'n_events': len(breath_events),
            'n_cycles': len(breath_cycles),
            'subband_energies': subband_energies,
            'gmm_params': self.gmm_params_record
        }


def visualize_detection(signal, sr, result, output_path=None):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    time = np.arange(len(signal)) / sr

    # 1. Waveform Plot
    axes[0].plot(time, signal, linewidth=0.5)
    axes[0].set_title('Denoised Signal Waveform', fontsize=20)        # ← 新增
    axes[0].set_xlabel('Time (s)', fontsize=16)                        # ← 新增
    axes[0].set_ylabel('Amplitude', fontsize=16)                       # ← 新增
    axes[0].tick_params(axis='both', labelsize=14)                     # ← 新增

    colors = {'inhalation': 'blue', 'exhalation': 'red'}
    for event in result['breath_events']:
        color = colors[event['phase']]
        axes[0].axvspan(event['start_time'], event['end_time'],
                        alpha=0.3, color=color, label=event['phase'])

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(),
                   loc='upper right', fontsize=14)                     # ← 新增

    # 2. Breathing Probability
    axes[1].plot(result['times'], result['breath_prob'], 'g-', linewidth=1)
    axes[1].fill_between(result['times'], 0, result['breath_prob'],
                         where=result['is_breathing'], alpha=0.3,
                         color='green', label='Breathing Region')
    axes[1].set_title('Breathing Probability (GMM)', fontsize=20)      # ← 新增
    axes[1].set_xlabel('Time (s)', fontsize=16)                        # ← 新增
    axes[1].set_ylabel('Probability', fontsize=16)                     # ← 新增
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='both', labelsize=14)                     # ← 新增
    axes[1].legend(loc='upper right', fontsize=14)                     # ← 新增

    # 3. Segmentation Results
    axes[2].fill_between(result['times'], 0, result['is_breathing'].astype(float),
                         alpha=0.5, color='green', step='mid')
    axes[2].set_title(
        f'Breathing Segmentation (Detected {result["n_events"]} Events, {result["n_cycles"]} Cycles)',
        fontsize=20)                                                    # ← 新增
    axes[2].set_xlabel('Time (s)', fontsize=16)                        # ← 新增
    axes[2].set_ylabel('Breath / Non-breath', fontsize=16)             # ← 新增
    axes[2].set_ylim([0, 1.2])
    axes[2].tick_params(axis='both', labelsize=14)                     # ← 新增

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def visualize_gmm_fitting(subband_energies, gmm_params, output_path=None):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 17  # ← 全局基础字体大小

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    subband_names = ['150-500Hz', '500-1000Hz', '1000-1500Hz', '1500-3000Hz']

    for i in range(4):
        ax = axes[i // 2, i % 2]
        data = subband_energies[:, i]
        params = gmm_params[i]

        ax.hist(data, bins=50, density=True, alpha=0.3, color='gray',
                label='Energy Distribution Histogram')

        x = np.linspace(data.min() - 2, data.max() + 2, 200)

        pdf_0 = stats.norm.pdf(x, params['mu_0'], np.sqrt(params['sigma_0']))
        pdf_1 = stats.norm.pdf(x, params['mu_1'], np.sqrt(params['sigma_1']))

        ax.plot(x, pdf_0, 'b--', lw=2, label='Background Noise (Non-breath)')
        ax.plot(x, pdf_1, 'r-', lw=2, label='Breathing Signal (Breath)')

        sqrt_s0 = np.sqrt(params['sigma_0'])
        sqrt_s1 = np.sqrt(params['sigma_1'])
        t_val = params['mu_0'] + (sqrt_s0 / (sqrt_s0 + sqrt_s1)) * (params['mu_1'] - params['mu_0'])

        ax.axvline(t_val, color='green', linestyle='-', lw=2,
                   label=f'Adaptive Threshold T={t_val:.2f}')

        ax.set_title(f'Sub-band: {subband_names[i]}', fontsize=19)  # ← 子图标题
        ax.set_xlabel('Log Energy', fontsize=17)                     # ← x轴标签
        ax.set_ylabel('Probability Density', fontsize=17)            # ← y轴标签
        ax.tick_params(axis='both', labelsize=14)                    # ← 刻度数字

        if i == 0:
            ax.legend(fontsize=14, loc='upper left')                 # ← 图例

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"GMM fitting plot saved to: {output_path}")
    else:
        plt.show()




if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    from scipy.io import wavfile

    # 注意：需要导入相关的类和函数
    # 例如：from your_module import BreathEventDetector, visualize_detection, visualize_gmm_fitting

    base_path = Path(__file__).parent
    output_dir = base_path / 'output'
    figure_dir = base_path / 'figures'

    subjects = [d for d in output_dir.iterdir() if d.is_dir()]

    if subjects:
        test_subject_dir = subjects[3]  # 假设测试第一个
        subject_name = test_subject_dir.name

        # --- 关键修改：确保受试者文件夹存在 ---
        subject_fig_dir = figure_dir / subject_name
        subject_fig_dir.mkdir(parents=True, exist_ok=True)

        wav_files = list(test_subject_dir.glob('*.wav')) + list(test_subject_dir.glob('*.WAV'))

        if wav_files:
            test_file = wav_files[0]

            # ========== 填充的数据处理逻辑开始 ==========
            print(f'=' * 40)
            print(f'Testing Subject: {subject_name}')
            print(f'File: {test_file.name}')
            print(f'=' * 40)

            # 2. 加载音频数据
            sr, data = wavfile.read(str(test_file))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            if len(data.shape) == 2:
                data = data[:, 0]
            # ========== 填充的数据处理逻辑结束 ==========

            detector = BreathEventDetector(sr=sr)
            result = detector.detect(data)

            # ========== 填充的打印结果逻辑 ==========
            print(f'\nDetection Results for {subject_name}:')
            print(f'  - Breath Events: {result["n_events"]}')
            print(f'  - Full Cycles: {result["n_cycles"]}')
            # =======================================

            # 4. 可视化输出 1: 存入专属文件夹
            detection_fig_path = str(subject_fig_dir / 'event_detection.png')
            visualize_detection(data, sr, result, detection_fig_path)
            print(f'✓ Detection plot saved to: {subject_name}/event_detection.png')

            # 5. 可视化输出 2: 存入专属文件夹
            if 'subband_energies' in result and 'gmm_params' in result:
                gmm_fig_path = str(subject_fig_dir / 'gmm_fitting.png')
                visualize_gmm_fitting(result['subband_energies'],
                                      result['gmm_params'],
                                      gmm_fig_path)
                print(f'✓ GMM fitting plot saved to: {subject_name}/gmm_fitting.png')
            else:
                print(f"\n[Warning] GMM data missing in results for {subject_name}.")
    else:
        print("No subject directories found in output folder.")