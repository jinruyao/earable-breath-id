# Earable User Identification using In-ear Respiration Audio Signal

基于耳内呼吸音频信号的耳戴式设备用户身份识别系统。

本项目实现了一套完整的呼吸音身份识别系统，包括信号预处理（语音检测、带通滤波、MODWT降噪）、呼吸事件检测（GMM自适应分割、时长检测、相位识别）、特征提取（BCD骨传导描述符 + RTD呼吸道描述符）以及用户识别（模板匹配 + SVM分类器）。

---

## 项目结构

```
├── denoise.py            # 预处理与降噪模块 (MODWT)
├── event_detection.py    # 呼吸事件检测模块 (GMM分割)
├── feature_extraction.py # 特征提取模块 (BCD + MFCC)
├── authentication.py     # 用户识别模块 (SVM + 模板匹配)
├── output/               # 处理后的音频和特征输出目录
├── figures/              # 可视化图表输出目录
├── models/               # 训练好的SVM模型
└── data/                 # 原始音频数据目录
```

---

## 环境要求

- Python 3.9+
- macOS / Linux / Windows

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 使用方法

### 1. 预处理与降噪

```python
from denoise import BreathSignDenoiser

denoiser = BreathSignDenoiser()
denoised_signal, info = denoiser.denoise(raw_signal, sr=16000)
```

### 2. 呼吸事件检测

```python
from event_detection import BreathEventDetector

detector = BreathEventDetector(sr=16000)
result = detector.detect(denoised_signal)
breath_cycles = result['breath_cycles']
```

### 3. 特征提取

```python
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor(sr=16000)
features, info = extractor.extract_from_all_cycles(breath_cycles, left_signal)
```

### 4. 训练 SVM 模型

```python
from authentication import SVMAuthenticator

svm_auth = SVMAuthenticator(kernel='rbf', C=10, gamma='scale')
svm_auth.train(features_by_user)
metrics = svm_auth.evaluate(test_data)
```

### 5. 完整流程

```bash
# 批量处理所有受试者
python denoise.py

# 提取特征
python feature_extraction.py

# 训练并评估 SVM
python authentication.py
```

---

## 实验结果

| 指标 | 数值 |
|------|------|
| 测试准确率 | 90.45% |
| 5折交叉验证准确率 | 89.07% ± 1.42% |
| 宏平均 F1 分数 | 0.8780 |
| 推理延迟 | 0.15ms |

相比模板匹配基线方法，准确率提升了 **48.88 个百分点**。

---

## 数据集

数据集包含 15 名受试者的骨传导呼吸录音，由导师预先采集。采样率 16kHz，16-bit 量化，双通道。

---

## 主要技术

| 模块 | 技术 |
|------|------|
| 降噪 | MODWT + 软阈值 |
| 事件检测 | GMM 自适应分割 + 时长检测 + 相位识别 |
| 特征提取 | BCD (SER+PCC) + MFCC (39维) |
| 分类 | RBF-SVM (C=10, gamma='scale') |
