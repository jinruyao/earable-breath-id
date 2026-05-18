# Earable User Identification using In-ear Respiration Audio Signal

基于耳内呼吸音频信号的耳戴式设备用户身份识别系统

This project implements a complete breath-based user identification system for earable devices, covering signal preprocessing (VAD, bandpass filtering, MODWT denoising), breath event detection (GMM-based adaptive segmentation, duration filtering, phase recognition), feature extraction (BCD bone-conduction descriptors + RTD respiratory tract descriptors), and user identification (template matching + SVM classifier).

本项目实现了一套完整的呼吸音身份识别系统，包括信号预处理（语音检测、带通滤波、MODWT降噪）、呼吸事件检测（GMM自适应分割、时长检测、相位识别）、特征提取（BCD骨传导描述符 + RTD呼吸道描述符）以及用户识别（模板匹配 + SVM分类器）。

---

## Project Structure / 项目结构

```
├── denoise.py            # Preprocessing & denoising module (MODWT) / 预处理与降噪模块
├── event_detection.py    # Breath event detection module (GMM) / 呼吸事件检测模块
├── feature_extraction.py # Feature extraction module (BCD + MFCC) / 特征提取模块
├── authentication.py     # User identification module (SVM + Template Matching) / 用户识别模块
├── output/               # Processed audio and feature output / 处理后的音频和特征输出
├── figures/              # Visualization output / 可视化图表输出
├── models/               # Trained SVM models / 训练好的SVM模型
└── data/                 # Raw audio data / 原始音频数据
```

---

## Requirements / 环境要求

- Python 3.9+
- macOS / Linux / Windows

---

## Installation / 安装依赖

```bash
pip install -r requirements.txt
```

---

## Usage / 使用方法

### 1. Preprocessing & Denoising / 预处理与降噪

```python
from denoise import BreathSignDenoiser

denoiser = BreathSignDenoiser()
denoised_signal, info = denoiser.denoise(raw_signal, sr=16000)
```

### 2. Breath Event Detection / 呼吸事件检测

```python
from event_detection import BreathEventDetector

detector = BreathEventDetector(sr=16000)
result = detector.detect(denoised_signal)
breath_cycles = result['breath_cycles']
```

### 3. Feature Extraction / 特征提取

```python
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sr=16000)
features, info = extractor.extract_from_all_cycles(breath_cycles, left_signal)
```

### 4. Train SVM Model / 训练 SVM 模型

```python
from authentication import SVMAuthenticator

svm_auth = SVMAuthenticator(kernel='rbf', C=10, gamma='scale')
svm_auth.train(features_by_user)
metrics = svm_auth.evaluate(test_data)
```

### 5. Full Pipeline / 完整流程

```bash
# Process all subjects / 批量处理所有受试者
python denoise.py

# Extract features / 提取特征
python feature_extraction.py

# Train and evaluate SVM / 训练并评估 SVM
python authentication.py
```

---

## Results / 实验结果

| Metric / 指标 | Value / 数值 |
|---|---|
| Test Accuracy / 测试准确率 | 90.45% |
| 5-Fold Cross-Validation Accuracy / 5折交叉验证准确率 | 89.07% ± 1.42% |
| Macro-average F1 Score / 宏平均 F1 分数 | 0.8780 |
| Inference Latency / 推理延迟 | 0.15ms |

Compared to the template matching baseline, accuracy improved by **48.88 percentage points**.

相比模板匹配基线方法，准确率提升了 **48.88 个百分点**。

---

## Dataset / 数据集

The dataset contains bone-conduction breath recordings from 15 subjects, pre-collected by the supervisor. Sampling rate: 16kHz, 16-bit quantization, dual-channel.

数据集包含 15 名受试者的骨传导呼吸录音，由导师预先采集。采样率 16kHz，16-bit 量化，双通道。

---

## Key Technologies / 主要技术

| Module / 模块 | Technology / 技术 |
|---|---|
| Denoising / 降噪 | MODWT + Soft Thresholding / 软阈值 |
| Event Detection / 事件检测 | GMM Adaptive Segmentation + Duration Filtering + Phase Recognition / GMM自适应分割 + 时长检测 + 相位识别 |
| Feature Extraction / 特征提取 | BCD (SER+PCC) + MFCC (39-dim / 39维) |
| Classification / 分类 | RBF-SVM (C=10, gamma='scale') |
