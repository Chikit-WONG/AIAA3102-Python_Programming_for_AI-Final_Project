"""
任务2：用户回复生成模型微调
输入：instruction + category信息
输出：response
"""

import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import get_model_tokenizer, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset

logger = get_logger()
seed_everything(42)

# ==================== 超参数配置 ====================
# 模型配置
model_id_or_path = 'models/Qwen2.5-1.5B-Instruct/' # Your model path here
system = 'You are a helpful customer service assistant. Generate appropriate responses to user requests based on their category.'
output_dir = 'checkpoint/task2_response_generation'

# 数据集配置
train_data_path = './assets/train.jsonl'  # 训练数据路径
max_length = 2048
num_proc = 4

# LoRA配置
lora_rank = 8
lora_alpha = 32

# 训练参数配置
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=100,
    eval_strategy='steps',
    eval_steps=100,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    metric_for_best_model='loss',
    save_total_limit=10,  # 保留所有检查点
    logging_steps=10,
    dataloader_num_workers=1,
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    greater_is_better=False,  # loss越小越好
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')

# ==================== 加载和处理数据 ====================
def load_jsonl_data(file_path):
    """加载JSONL格式的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data

def create_response_generation_dataset(data):
    """
    创建回复生成任务数据集
    将数据转换为messages格式（ms-swift要求的格式）
    """
    formatted_data = []
    for item in data:
        # 构造用户消息和助手回复
        user_message = f"This is a Type {item['category']} user request: {item['instruction']}. Please formulate an appropriate response."
        assistant_message = item['response']
        
        formatted_item = {
            'messages': [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
        formatted_data.append(formatted_item)
    
    return Dataset.from_list(formatted_data)

# 加载训练数据
logger.info(f'Loading training data from {train_data_path}')
train_data = load_jsonl_data(train_data_path)
logger.info(f'Total training samples: {len(train_data)}')

# 加载验证数据（使用独立的validation.jsonl）
val_data_path = '/hpc2hdd/home/yuxuanzhao/haodong/3102project/assets/validation.jsonl'
if os.path.exists(val_data_path):
    logger.info(f'Loading validation data from {val_data_path}')
    val_data = load_jsonl_data(val_data_path)
    logger.info(f'Validation samples: {len(val_data)}')
else:
    # 如果没有独立的验证集，从训练集中划分
    logger.warning(f'Validation file {val_data_path} not found, splitting from training data')
    split_ratio = 0.1
    split_idx = int(len(train_data) * (1 - split_ratio))
    val_data = train_data[split_idx:]
    train_data = train_data[:split_idx]
    logger.info(f'Training samples after split: {len(train_data)}')
    logger.info(f'Validation samples after split: {len(val_data)}')

# 创建数据集
train_dataset = create_response_generation_dataset(train_data)
val_dataset = create_response_generation_dataset(val_data)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'Sample data: {train_dataset[0]}')

# ==================== 加载模型和模板 ====================
model, tokenizer = get_model_tokenizer(model_id_or_path)
logger.info(f'model_info: {model.model_info}')

template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
template.set_mode('train')

# 添加LoRA层
target_modules = find_all_linears(model)
lora_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=target_modules
)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# 打印模型信息
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

# ==================== 编码数据 ====================
train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# 打印样本
template.print_inputs(train_dataset[0])

# ==================== 开始训练 ====================
model.enable_input_require_grads()

# 配置早停回调
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # 连续3次评估没有改善则停止
    early_stopping_threshold=0.0  # 改善的最小阈值
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
    callbacks=[early_stopping_callback]  # 添加早停回调
)

logger.info('Starting training for Task 2: Response Generation Model')
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
best_model_checkpoint = trainer.state.best_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
logger.info(f'best_model_checkpoint: {best_model_checkpoint}')

# 保存最终模型
final_output_dir = os.path.join(output_dir, 'final_model')
trainer.save_model(final_output_dir)
logger.info(f'Final model saved to: {final_output_dir}')
logger.info('Training completed for Task 2: Response Generation Model')