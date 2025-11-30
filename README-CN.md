# AIAA3102 Python Programming for AI - Final Project

## 📋 项目简介

本项目是 AIAA3102 课程的期末项目，专注于**客户服务意图分类与响应生成**任务。项目使用 Qwen2.5 大语言模型，通过监督微调（SFT）技术，实现对客户支持对话的智能处理，并与基线方法（0-shot、Few-shot）进行对比实验。

## 🗂️ 项目结构

```
.    
├── training/                      # 模型训练模块
│   ├── classification_sft.py      # 意图分类任务 SFT 训练
│   ├── response_generation_sft.py # 响应生成任务 SFT 训练
│   ├── run_benchmark.py           # 基准测试运行脚本
│   ├── benchmark_utils.py         # 基准测试工具函数
│   └── visualize_results.py       # 结果可视化脚本
│
├── eval/                          # 模型评估模块
│   ├── classification_eval.py     # 意图分类评估
│   ├── classification_eval_robust.py # 鲁棒性分类评估
│   ├── response_generation_eval. py   # 响应生成评估
│   ├── config.json                # 评估配置文件
│   └── prompt.json                # Prompt 模板配置
│
├── infer/                         # 微调模型推理模块
│   ├── task1_inference.py         # 任务1（意图分类）推理
│   └── task2_inference.py         # 任务2（响应生成）推理
│
├── comparison_infer/              # 基线对比推理模块
│   ├── task1_baseline_0shot.py    # 任务1 基线推理（0-shot）
│   ├── task1_baseline_10shot.py   # 任务1 基线推理（10-shot）
│   ├── task2_baseline_0shot.py    # 任务2 基线推理（0-shot）
│   └── task2_baseline_10shot.py   # 任务2 基线推理（10-shot）
│
├── assets/                        # 数据集文件目录
│   ├── train. jsonl                # 训练集
│   ├── validation.jsonl           # 验证集
│   └── test.jsonl                 # 测试集
│
├── output/                        # 推理预测结果目录
│   ├── task1_predictions.jsonl           # 任务1 SFT模型预测结果
│   ├── task1_baseline_0shot_predictions.jsonl   # 任务1 0-shot预测结果
│   ├── task1_baseline_10shot_predictions.jsonl  # 任务1 10-shot预测结果
│   ├── task2_predictions.jsonl           # 任务2 SFT模型预测结果
│   ├── task2_baseline_0shot_predictions.jsonl   # 任务2 0-shot预测结果
│   └── task2_baseline_10shot_predictions.jsonl  # 任务2 10-shot预测结果
│
├── checkpoint/                    # 模型 checkpoint 目录（本地生成，未上传）
│   ├── task1_classification/      # 任务1 分类模型权重
│   └── task2_response_generation/ # 任务2 响应生成模型权重
│
├── eval_output/                   # 评估输出目录
├── benchmark_output/              # 基准测试输出目录
│
├── app.py                         # 🌐 Demo 后端 API 服务
├── index.html                     # 🌐 Demo 前端页面
│
├── split. py                       # 数据集分层抽样划分脚本
├── tokenize.py                    # Qwen2.5 分词预处理脚本
├── test. py                        # 测试脚本
│
├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv  # 原始数据集
├── data. jsonl                     # 处理后的数据
└── data_tokenize.jsonl            # 分词后的数据
```

## 📊 数据集

本项目使用 **Bitext Customer Support Training Dataset**，包含约 27K 条客户服务对话样本，涵盖 11 个意图类别：

`FEEDBACK`, `REFUND`, `ACCOUNT`, `ORDER`, `SHIPPING`, `SUBSCRIPTION`, `PAYMENT`, `CANCEL`, `DELIVERY`, `INVOICE`, `CONTACT`

### 数据集划分

数据集已划分为三部分，存储在 `assets/` 目录下：

| 文件 | 说明 |
|------|------|
| `train.jsonl` | 训练集 |
| `validation.jsonl` | 验证集 |
| `test.jsonl` | 测试集 |

### 数据格式

每条数据为 JSON 格式，包含以下字段：

```json
{
  "flags": "BLM",
  "instruction": "用户输入文本",
  "category": "ORDER",
  "intent": "place_order",
  "response": "客服回复文本..."
}
```

| 字段 | 说明 |
|------|------|
| `flags` | 数据标记 |
| `instruction` | 用户输入/查询 |
| `category` | 意图大类（11 类） |
| `intent` | 细分意图 |
| `response` | 期望的客服回复 |

### 数据预处理

1.  **数据划分**：使用 `split.py` 进行分层抽样，确保训练集和验证集中各类别比例一致
   ```bash
   python split.py --input_file data.jsonl --train_file train.jsonl --val_file val.jsonl --test_ratio 0.1
   ```

2. **数据分词**：使用 `tokenize. py` 对数据进行 Qwen2. 5 分词预处理
   ```bash
   python tokenize.py data.jsonl data_tokenize.jsonl --model Qwen/Qwen2.5-7B-Instruct
   ```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch
- Transformers
- Swift (ModelScope)
- Flask（Demo 服务）
- tqdm

### 安装依赖

```bash
pip install torch transformers tqdm ms-swift flask flask-cors
```
或
```bash
pip install -r requirements.txt
```

## 📝 任务说明

### 任务一：意图分类

根据用户输入预测客户服务意图类别（11 类分类任务）

#### SFT 微调方法
```bash
# 训练
python training/classification_sft.py

# 推理
python infer/task1_inference.py

# 评估
python eval/classification_eval.py
```

#### 基线对比方法
```bash
# 0-shot（无示例）
python comparison_infer/task1_baseline_0shot.py

# 10-shot（每类1个示例，共11个）
python comparison_infer/task1_baseline_10shot. py
```

### 任务二：响应生成

根据用户问题和类别生成合适的客服回复

#### SFT 微调方法
```bash
# 训练
python training/response_generation_sft.py

# 推理
python infer/task2_inference.py

# 评估
python eval/response_generation_eval.py
```

#### 基线对比方法
```bash
# 0-shot（无示例）
python comparison_infer/task2_baseline_0shot.py

# 10-shot（每类1个示例，共11个）
python comparison_infer/task2_baseline_10shot.py
```

## 🔬 实验对比

本项目对比了三种方法的性能：

| 方法 | 描述 | 任务1脚本 | 任务2脚本 |
|------|------|----------|----------|
| **0-shot** | 原始模型，无示例 | `task1_baseline_0shot.py` | `task2_baseline_0shot.py` |
| **Few-shot (10-shot)** | 原始模型 + 平衡采样示例（每类1个） | `task1_baseline_10shot.py` | `task2_baseline_10shot.py` |
| **SFT 微调** | 监督微调后的模型 | `task1_inference.py` | `task2_inference. py` |

### Few-shot 策略说明

Few-shot 基线采用**平衡抽样策略**：
- 从验证集中每个类别各选择 1 个样本作为示例
- 共 11 个类别，因此总计 11 个示例
- 使用固定随机种子（seed=42）确保可复现性

## 📈 基准测试

运行完整的基准测试：
```bash
python training/run_benchmark.py
```

可视化结果：
```bash
python training/visualize_results. py
```

## 🔧 配置说明

| 配置项 | 位置 | 说明 |
|--------|------|------|
| 评估配置 | `eval/config.json` | 评估参数设置 |
| Prompt 模板 | `eval/prompt.json` | 各任务的 Prompt 模板 |
| 模型路径 | 各脚本内 | 需根据实际环境修改 |
| Demo 配置 | `app.py` | Demo 服务参数配置 |

## 📁 输出文件

### `output/` - 推理预测结果

| 文件 | 说明 |
|------|------|
| `task1_predictions. jsonl` | 任务1 SFT 微调模型预测结果 |
| `task1_baseline_0shot_predictions.jsonl` | 任务1 0-shot 基线预测结果 |
| `task1_baseline_10shot_predictions. jsonl` | 任务1 10-shot 基线预测结果 |
| `task2_predictions.jsonl` | 任务2 SFT 微调模型预测结果 |
| `task2_baseline_0shot_predictions. jsonl` | 任务2 0-shot 基线预测结果 |
| `task2_baseline_10shot_predictions.jsonl` | 任务2 10-shot 基线预测结果 |

#### 预测结果格式

**任务1（意图分类）预测结果：**
```json
{
  "instruction": "用户输入文本",
  "predicted_category": "ORDER",
  "ground_truth_category": "ORDER",
  "ground_truth_response": "期望的客服回复..."
}
```

**任务2（响应生成）预测结果：**
```json
{
  "instruction": "用户输入文本",
  "category": "ORDER",
  "generated_response": "模型生成的回复.. .",
  "ground_truth_response": "期望的客服回复..."
}
```

### `checkpoint/` - 模型 Checkpoint

> ⚠️ **注意**：由于 checkpoint 文件较大，未上传至 GitHub 仓库。本地运行训练脚本时会自动生成。

| 目录 | 说明 |
|------|------|
| `task1_classification/` | 意图分类任务的微调模型权重 |
| `task2_response_generation/` | 响应生成任务的微调模型权重 |

### 其他输出目录

| 目录 | 内容 |
|------|------|
| `eval_output/` | 评估结果和指标 |
| `benchmark_output/` | 基准测试结果 |

## 🌐 Demo 演示

本项目提供了一个完整的 Web Demo，可以直观地体验智能邮件助手的功能。

### 功能特性

- 🏷️ **邮件分类**：自动识别用户邮件的意图类别
- 💬 **回复生成**：根据邮件内容和分类生成专业的客服回复
- 🔄 **一键处理**：支持同时进行分类和回复生成

### 启动 Demo

```bash
# 安装额外依赖
pip install flask flask-cors

# 启动服务
python app.py
```

### 访问地址

服务启动后，可通过以下方式访问：

| 访问方式 | 地址 |
|---------|------|
| 本地访问 | http://localhost:5000 |
| 局域网访问 | http://<服务器IP>:5000 |

### API 接口

| 方法 | 接口 | 说明 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/classify` | 邮件分类 |
| POST | `/api/generate` | 回复生成 |
| POST | `/api/process` | 完整处理（分类+生成） |

#### 接口请求示例

**邮件分类 `/api/classify`**
```json
// 请求
{"email": "I want to cancel my subscription"}

// 响应
{"success": true, "category": "CANCEL"}
```

**回复生成 `/api/generate`**
```json
// 请求
{"email": "I want to cancel my subscription", "category": "CANCEL"}

// 响应
{"success": true, "response": "We're sorry to hear that you want to cancel... "}
```

**完整处理 `/api/process`**
```json
// 请求
{"email": "I want to cancel my subscription"}

// 响应
{
  "success": true,
  "category": "CANCEL",
  "response": "We're sorry to hear that you want to cancel..."
}
```

### Demo 配置说明

Demo 服务的配置参数位于 `app. py` 文件顶部：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `HOST` | `0.0.0. 0` | 服务监听地址 |
| `PORT` | `5000` | 服务端口 |
| `BASE_MODEL_PATH` | - | Qwen2.5 基础模型路径 |
| `TASK1_CHECKPOINT` | `checkpoint/task1_classification/final_model` | 分类模型路径 |
| `TASK2_CHECKPOINT` | `checkpoint/task2_response_generation/final_model` | 生成模型路径 |

## 📄 License

本项目仅供学术研究使用。

## 👥 作者

AIAA3102 课程学生项目