# denoise.py

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import pywt
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Matplotlib 中文字体配置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False



class BreathSignDenoiser:

    def __init__(self, low_freq=100, high_freq=3000,
                 wavelet='db4', wavelet_level=4, threshold_mode='soft',
                 threshold_rule='universal', rescaling='sln'):
        self.low_freq = low_freq
        self.high_freq = high_freq

        # 默认参数 (GA禁用时使用，或作为初始值)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.threshold_mode = threshold_mode
        self.threshold_rule = threshold_rule
        self.rescaling = rescaling

    def load_audio(self, filepath):
        sr, data = wavfile.read(filepath)

        # 归一化到[-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0

        # 分离通道
        if len(data.shape) == 2:
            return sr, data[:, 0], data[:, 1]
        else:
            return sr, data, None

    def save_audio(self, filepath, sr, left_channel, right_channel=None):
        if right_channel is not None:
            data = np.column_stack([left_channel, right_channel])
        else:
            data = left_channel

        data = np.clip(data, -1, 1)
        data = (data * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

    def speaking_detection(self, signal, sr, frame_size=0.025, hop_size=0.01,
                          psd_threshold=None):
        frame_len = int(frame_size * sr)
        hop_len = int(hop_size * sr)

        # 分帧计算PSD
        n_frames = (len(signal) - frame_len) // hop_len + 1
        psd_values = []

        for i in range(n_frames):
            start = i * hop_len
            frame = signal[start:start + frame_len]
            # PSD估计: 帧内平均功率
            psd = np.mean(frame ** 2)
            psd_values.append(psd)

        psd_values = np.array(psd_values)

        # 自适应阈值: median + 2*std (鲁棒统计量)
        if psd_threshold is None:
            psd_threshold = np.median(psd_values) + 2 * np.std(psd_values)

        valid_frames = psd_values < psd_threshold

        # 帧级掩码扩展到样本级
        sample_mask = np.ones(len(signal), dtype=bool)
        for i, valid in enumerate(valid_frames):
            if not valid:
                start = i * hop_len
                end = min(start + frame_len, len(signal))
                sample_mask[start:end] = False

        return sample_mask, psd_values, psd_threshold

    def bandpass_filter(self, signal, sr, order=4):
        nyq = sr / 2
        low = self.low_freq / nyq
        high = min(self.high_freq / nyq, 0.99)  # 防止超过Nyquist频率

        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)

        return filtered

    def modwt(self, signal, wavelet, level):
        n = len(signal)
        wav = pywt.Wavelet(wavelet)

        # MODWT滤波器: 标准小波滤波器除以√2
        g = np.array(wav.dec_lo) / np.sqrt(2)  # 低通(尺度)滤波器
        h = np.array(wav.dec_hi) / np.sqrt(2)  # 高通(小波)滤波器
        L = len(g)

        details = []
        V = signal.astype(np.float64)

        for j in range(1, level + 1):
            # 第j层的上采样步长
            step = 2 ** (j - 1)

            # 向量化实现: 构建循环索引矩阵
            t_indices = np.arange(n)
            l_indices = np.arange(L)
            idx_matrix = (t_indices[:, None] - l_indices[None, :] * step) % n

            V_shifted = V[idx_matrix]

            # 细节系数(高通滤波)
            W = np.dot(V_shifted, h)
            # 近似系数(低通滤波)
            V_new = np.dot(V_shifted, g)

            details.append(W)
            V = V_new

        # 返回格式与pywt.wavedec一致: [cA_J, cD_J, cD_{J-1}, ..., cD_1]
        return [V] + details[::-1]

    def imodwt(self, coeffs, wavelet):
        wav = pywt.Wavelet(wavelet)

        # 逆变换用重构滤波器，并除以√2（MODWT归一化）
        g = np.array(wav.rec_lo) / 2
        h = np.array(wav.rec_hi) / 2
        L = len(g)

        level = len(coeffs) - 1
        n = len(coeffs[0])

        V = coeffs[0].astype(np.float64)

        for j in range(level, 0, -1):
            W = coeffs[level - j + 1]
            step = 2 ** (j - 1)

            t_indices = np.arange(n)
            l_indices = np.arange(L)

            # 逆变换索引方向：向前偏移（+方向）
            idx_matrix = (t_indices[:, None] + l_indices[None, :] * step) % n

            V_new = np.dot(V[idx_matrix], g) + np.dot(W[idx_matrix], h)
            V = V_new

        return V

    def _calculate_threshold(self, coeffs, n):
        """
        根据当前参数计算阈值

        Args:
            coeffs: 小波系数
            n: 信号长度

        Returns:
            threshold: 计算的阈值
        """
        # 噪声估计 (使用最高频细节系数的MAD)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745

        # 根据重缩放方法调整sigma
        if self.rescaling == 'sln':
            pass  # 单层噪声估计
        elif self.rescaling == 'mln':
            # 多层噪声估计
            sigmas = []
            for i in range(1, len(coeffs)):
                s = np.median(np.abs(coeffs[i])) / 0.6745
                sigmas.append(s)
            sigma = np.mean(sigmas)
        elif self.rescaling == 'one':
            # 全局噪声估计
            all_details = np.concatenate([coeffs[i] for i in range(1, len(coeffs))])
            sigma = np.median(np.abs(all_details)) / 0.6745

        # 根据阈值规则计算阈值
        if self.threshold_rule == 'universal':
            threshold = sigma * np.sqrt(2 * np.log(n))
        elif self.threshold_rule == 'sqtwolog':
            threshold = sigma * np.sqrt(2 * np.log(n)) / np.sqrt(np.log(n + 1))
        elif self.threshold_rule == 'minimaxi':
            if n > 32:
                threshold = sigma * (0.3936 + 0.1829 * np.log2(n))
            else:
                threshold = 0
        else:
            threshold = sigma * np.sqrt(2 * np.log(n))

        return threshold

    def modwt_denoise(self, signal):
        """
        Step 3: 基于固定参数的 MODWT 小波降噪
        """
        # 1. MODWT分解 (使用初始化时确定的 wavelet 和 level)
        coeffs = self.modwt(signal, self.wavelet, self.wavelet_level)

        # 2. 计算阈值 (使用初始化时确定的 threshold_rule 和 rescaling)
        threshold = self._calculate_threshold(coeffs, len(signal))

        # 3. 阈值处理
        denoised_coeffs = [coeffs[0]] # 保留近似系数
        for i in range(1, len(coeffs)):
            denoised = pywt.threshold(coeffs[i], threshold, mode=self.threshold_mode)
            denoised_coeffs.append(denoised)

        # 4. MODWT逆变换重构
        return self.imodwt(denoised_coeffs, self.wavelet)

    def denoise(self, signal, sr, apply_speaking_detection=True):
        """
        完整降噪流程

        按顺序执行三阶段降噪处理:
            Step 1: Speaking Detection → Step 2: Band-pass Filter → Step 3: MODWT Denoise

        Args:
            signal: 原始音频信号
            sr: 采样率(Hz)
            apply_speaking_detection: 是否执行语音检测步骤

        Returns:
            denoised: 降噪后信号
            info: 处理信息字典，包含语音检测统计等
        """
        info = {}

        # Step 1: Speaking Detection
        if apply_speaking_detection:
            mask, psd_values, psd_threshold = self.speaking_detection(signal, sr)
            speech_ratio = 1 - np.mean(mask)
            info['speech_ratio'] = speech_ratio
            info['psd_threshold'] = psd_threshold

            if speech_ratio > 0.5:
                print(f"  Warning: {speech_ratio*100:.1f}% frames detected as speech")

        # Step 2: Band-pass Filtering
        filtered = self.bandpass_filter(signal, sr)

        # Step 3: MODWT Denoising
        denoised = self.modwt_denoise(filtered)

        return denoised, info

    def process_file(self, input_path, output_path):
        """
        处理单个音频文件

        Args:
            input_path: 输入WAV文件路径
            output_path: 输出WAV文件路径

        Returns:
            处理结果字典，包含采样率、通道数、时长等信息
        """
        sr, left, right = self.load_audio(input_path)

        left_denoised, info_left = self.denoise(left, sr)

        right_denoised = None
        if right is not None:
            right_denoised, _ = self.denoise(right, sr)

        self.save_audio(output_path, sr, left_denoised, right_denoised)

        return {
            'sr': sr,
            'stereo': right is not None,
            'duration': len(left) / sr,
            'info_left': info_left
        }


def batch_process(input_dir, output_dir, denoiser):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    subjects = [d for d in input_path.iterdir()
                if d.is_dir() and not d.name.startswith('.')
                and d.name != 'breathsign_denoise']

    results = {}
    total_files = 0

    for subject_dir in sorted(subjects):
        subject_name = subject_dir.name
        print(f"\nProcessing subject: {subject_name}")

        subject_output = output_path / subject_name
        subject_output.mkdir(parents=True, exist_ok=True)

        wav_files = list(subject_dir.glob('*.wav')) + list(subject_dir.glob('*.WAV'))

        subject_results = []
        for wav_file in sorted(wav_files):
            output_file = subject_output / wav_file.name

            try:
                result = denoiser.process_file(str(wav_file), str(output_file))
                subject_results.append({
                    'file': wav_file.name,
                    'status': 'success',
                    **result
                })
                print(f"  ✓ {wav_file.name}")
            except Exception as e:
                subject_results.append({
                    'file': wav_file.name,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"  ✗ {wav_file.name}: {e}")

        results[subject_name] = subject_results
        total_files += len(wav_files)

    return results, total_files


def visualize_comparison(original_path, denoised_path, output_figure_path):
    sr_orig, orig = wavfile.read(original_path)
    sr_den,  den  = wavfile.read(denoised_path)

    if orig.dtype == np.int16:
        orig = orig.astype(np.float32) / 32768.0
    if den.dtype == np.int16:
        den  = den.astype(np.float32)  / 32768.0

    if len(orig.shape) == 2: orig = orig[:, 0]
    if len(den.shape)  == 2: den  = den[:,  0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    time = np.arange(len(orig)) / sr_orig

    # 1. Original Waveform
    axes[0, 0].plot(time, orig, linewidth=0.5)
    axes[0, 0].set_title('Original Signal - Time Domain', fontsize=20)   # ← 新增
    axes[0, 0].set_xlabel('Time (s)',  fontsize=16)                        # ← 新增
    axes[0, 0].set_ylabel('Amplitude', fontsize=16)                        # ← 新增
    axes[0, 0].tick_params(axis='both', labelsize=14)                      # ← 新增
    axes[0, 0].set_xlim([0, time[-1]])

    # 2. Denoised Waveform
    time_den = np.arange(len(den)) / sr_den
    axes[0, 1].plot(time_den, den, linewidth=0.5, color='green')
    axes[0, 1].set_title('Denoised Signal - Time Domain', fontsize=20)    # ← 新增
    axes[0, 1].set_xlabel('Time (s)',  fontsize=16)                        # ← 新增
    axes[0, 1].set_ylabel('Amplitude', fontsize=16)                        # ← 新增
    axes[0, 1].tick_params(axis='both', labelsize=14)                      # ← 新增
    axes[0, 1].set_xlim([0, time_den[-1]])

    # 3. Original Spectrum
    freqs_orig = np.fft.rfftfreq(len(orig), 1 / sr_orig)
    fft_orig   = np.abs(np.fft.rfft(orig))
    axes[1, 0].semilogy(freqs_orig, fft_orig, linewidth=0.5)
    axes[1, 0].set_title('Original Signal - Spectrum', fontsize=20)       # ← 新增
    axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=16)                   # ← 新增
    axes[1, 0].set_ylabel('Magnitude',      fontsize=16)                   # ← 新增
    axes[1, 0].tick_params(axis='both', labelsize=14)                      # ← 新增
    axes[1, 0].set_xlim([0, 5000])
    axes[1, 0].axvline(x=100,  color='r', linestyle='--', alpha=0.5, label='100Hz Cutoff')
    axes[1, 0].axvline(x=3000, color='r', linestyle='--', alpha=0.5, label='3kHz Cutoff')
    axes[1, 0].legend(fontsize=14)                                         # ← 新增

    # 4. Denoised Spectrum
    freqs_den = np.fft.rfftfreq(len(den), 1 / sr_den)
    fft_den   = np.abs(np.fft.rfft(den))
    axes[1, 1].semilogy(freqs_den, fft_den, linewidth=0.5, color='green')
    axes[1, 1].set_title('Denoised Signal - Spectrum', fontsize=20)       # ← 新增
    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=16)                   # ← 新增
    axes[1, 1].set_ylabel('Magnitude',      fontsize=16)                   # ← 新增
    axes[1, 1].tick_params(axis='both', labelsize=14)                      # ← 新增
    axes[1, 1].set_xlim([0, 5000])
    axes[1, 1].axvline(x=100,  color='r', linestyle='--', alpha=0.5, label='100Hz Cutoff')
    axes[1, 1].axvline(x=3000, color='r', linestyle='--', alpha=0.5, label='3kHz Cutoff')
    axes[1, 1].legend(fontsize=14)                                         # ← 新增

    plt.tight_layout()
    plt.savefig(output_figure_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fig



def main():
    base_dir = Path(__file__).parent
    input_dir = base_dir / 'data'
    output_dir = Path(__file__).parent / 'output'
    figure_dir = Path(__file__).parent / 'figures'

    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    denoiser = BreathSignDenoiser(
        low_freq=100, high_freq=3000,
        wavelet='db4', wavelet_level=4,
        threshold_mode='soft', threshold_rule='universal'
    )

    # ... [此处省略 Phase 1 打印信息] ...

    results, total_files = batch_process(input_dir, output_dir, denoiser)

    print("\nGenerating visualizations into subject folders...")
    for subject_name, subject_results in results.items():
        success_files = [r for r in subject_results if r['status'] == 'success']
        if success_files:
            # --- 关键修改：创建受试者专属图片文件夹 ---
            subject_fig_dir = figure_dir / subject_name
            subject_fig_dir.mkdir(parents=True, exist_ok=True)

            sample_file = success_files[0]['file']
            original_path = input_dir / subject_name / sample_file
            denoised_path = output_dir / subject_name / sample_file

            # 修改路径：存入专属文件夹
            figure_path = subject_fig_dir / f"comparison.png"

            try:
                visualize_comparison(str(original_path), str(denoised_path), str(figure_path))
                print(f"  ✓ Saved to: {subject_name}/comparison.png")
            except Exception as e:
                print(f"  ✗ {subject_name}: {e}")

    print("\nDone!")

if __name__ == '__main__':
    main()