"""
LLM微调效率对比Benchmark
对比全精度、LoRA、4-bit QLoRA、8-bit QLoRA的训练时间和内存占用
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict, Any

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from swift.llm import get_model_tokenizer, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
from datasets import Dataset

from benchmark_utils import (
    GPUMemoryMonitor, BenchmarkCallback, BenchmarkResults,
    TrainingMetrics, get_system_info
)

logger = get_logger()
seed_everything(42)


class TrainingBenchmark:
    """训练基准测试类"""
    
    def __init__(
        self,
        model_path: str,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        output_base_dir: str = "./benchmark_output",
        max_samples: int = 500,  # 限制样本数量以加快benchmark
        num_epochs: int = 1,  # benchmark只训练1个epoch
    ):
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.output_base_dir = output_base_dir
        self.max_samples = max_samples
        self.num_epochs = num_epochs
        
        self.benchmark_results = BenchmarkResults(
            output_dir=os.path.join(output_base_dir, "results")
        )
        
        # 获取系统信息
        self.system_info = get_system_info()
        logger.info(f"System Info: {json.dumps(self.system_info, indent=2)}")
    
    def load_and_prepare_data(self, task_type: str = "classification") -> tuple:
        """加载和准备数据"""
        logger.info(f"Loading data from {self.train_data_path}")
        
        # 加载训练数据
        train_data = []
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                line = line.strip()
                if line:
                    train_data.append(json.loads(line))
        
        # 加载验证数据
        val_data = []
        val_path = self.val_data_path if self.val_data_path else self.train_data_path
        with open(val_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= min(100, self.max_samples // 5):  # 验证集用更少的数据
                    break
                line = line.strip()
                if line:
                    val_data.append(json.loads(line))
        
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # 转换为Dataset格式
        if task_type == "classification":
            train_dataset = self._create_classification_dataset(train_data)
            val_dataset = self._create_classification_dataset(val_data)
        else:
            train_dataset = self._create_response_dataset(train_data)
            val_dataset = self._create_response_dataset(val_data)
        
        return train_dataset, val_dataset
    
    def _create_classification_dataset(self, data):
        """创建分类任务数据集"""
        formatted_data = []
        for item in data:
            user_message = f"Please classify the following user request into the appropriate category: {item['instruction']}"
            assistant_message = item['category']
            
            formatted_item = {
                'messages': [
                    {'role': 'user', 'content': user_message},
                    {'role': 'assistant', 'content': assistant_message}
                ]
            }
            formatted_data.append(formatted_item)
        
        return Dataset.from_list(formatted_data)
    
    def _create_response_dataset(self, data):
        """创建回复生成任务数据集"""
        formatted_data = []
        for item in data:
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
    
    def run_benchmark(
        self,
        config_name: str,
        train_type: str = "lora",  # 'full', 'lora'
        quantization: Optional[str] = None,  # None, '4bit', '8bit'
        lora_rank: int = 8,
        learning_rate: float = 1e-4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        task_type: str = "classification"
    ):
        """运行单个配置的benchmark"""
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Starting benchmark: {config_name}")
        logger.info(f"Config: train_type={train_type}, quantization={quantization}, lora_rank={lora_rank}")
        logger.info(f"{'='*100}\n")
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 设置输出目录
        output_dir = os.path.join(self.output_base_dir, config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化内存监控
        memory_monitor = GPUMemoryMonitor(device=0)
        memory_monitor.reset()
        
        # 加载数据
        train_dataset, val_dataset = self.load_and_prepare_data(task_type=task_type)
        
        # 配置量化
        model_kwargs = {'device_map': 'auto'}
        if quantization == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True
            )
            model_kwargs['quantization_config'] = quantization_config
            torch_dtype = torch.bfloat16
        elif quantization == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=['lm_head'],
                llm_int8_threshold=6.0
            )
            model_kwargs['quantization_config'] = quantization_config
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.bfloat16
        
        # 加载模型
        logger.info("Loading model...")
        memory_monitor.capture_snapshot()
        
        model, tokenizer = get_model_tokenizer(
            self.model_path,
            torch_dtype=torch_dtype,
            model_kwargs=model_kwargs
        )
        
        memory_monitor.capture_snapshot()
        logger.info(f"Model loaded. Memory: {memory_monitor.get_peak_memory():.2f}GB")
        
        # 配置模板
        system_prompt = 'You are a helpful assistant specialized in classifying user requests.' if task_type == "classification" else 'You are a helpful customer service assistant.'
        template = get_template(
            model.model_meta.template,
            tokenizer,
            default_system=system_prompt,
            max_length=2048
        )
        template.set_mode('train')
        
        # 应用训练策略
        if train_type == 'lora' or quantization:
            # LoRA或QLoRA配置
            target_modules = find_all_linears(model)
            lora_config = LoraConfig(
                task_type='CAUSAL_LM',
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=target_modules
            )
            model = Swift.prepare_model(model, lora_config)
            logger.info(f"LoRA applied. Config: {lora_config}")
            
            # 调整学习率（QLoRA通常需要更高的学习率）
            if quantization:
                learning_rate = learning_rate * 2
        
        # 打印模型信息
        model_parameter_info = get_model_parameter_info(model)
        logger.info(f"Model parameter info: {model_parameter_info}")
        
        memory_monitor.capture_snapshot()
        
        # 编码数据
        logger.info("Encoding datasets...")
        train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=4)
        val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=4)
        
        # 训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=0.1,
            lr_scheduler_type='cosine',
            warmup_ratio=0.05,
            num_train_epochs=self.num_epochs,
            logging_steps=5,
            eval_strategy='steps',
            eval_steps=50,
            save_strategy='steps',
            save_steps=100,
            report_to=['tensorboard'],
            logging_first_step=True,
            dataloader_num_workers=1,
            bf16=(torch_dtype == torch.bfloat16),
            fp16=(torch_dtype == torch.float16),
        )
        
        # 初始化回调
        benchmark_callback = BenchmarkCallback(
            memory_monitor=memory_monitor,
            log_every_n_steps=10
        )
        
        # 创建trainer
        model.enable_input_require_grads()
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=template.data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            template=template,
            callbacks=[benchmark_callback]
        )
        
        # 开始训练
        logger.info(f"Starting training for {config_name}...")
        train_result = trainer.train()
        
        # 收集指标
        total_training_time = benchmark_callback.get_total_training_time()
        avg_step_time = benchmark_callback.get_avg_step_time()
        
        # 计算吞吐量
        total_samples = len(train_dataset) * self.num_epochs
        samples_per_second = total_samples / total_training_time if total_training_time > 0 else 0
        
        # 获取最终损失
        final_train_loss = train_result.training_loss
        
        # 评估
        eval_result = trainer.evaluate()
        final_eval_loss = eval_result.get('eval_loss', 0.0)
        
        # 创建指标对象
        metrics = TrainingMetrics(
            config_name=config_name,
            model_name=os.path.basename(self.model_path),
            train_type=train_type,
            quantization=quantization,
            total_training_time=total_training_time,
            avg_step_time=avg_step_time,
            samples_per_second=samples_per_second,
            peak_memory_gb=memory_monitor.get_peak_memory(),
            avg_memory_gb=memory_monitor.get_avg_memory(),
            memory_snapshots=[vars(s) for s in memory_monitor.snapshots[-10:]],  # 保存最后10个快照
            final_train_loss=final_train_loss,
            final_eval_loss=final_eval_loss,
            total_steps=trainer.state.global_step,
            gpu_name=self.system_info.get('gpu_name', 'Unknown'),
            gpu_total_memory_gb=self.system_info.get('gpu_total_memory_gb', 0),
            torch_version=self.system_info.get('torch_version', 'Unknown'),
            cuda_version=self.system_info.get('cuda_version', 'Unknown'),
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lora_rank=lora_rank if train_type != 'full' else None,
            learning_rate=learning_rate,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 保存结果
        self.benchmark_results.add_result(metrics)
        
        # 清理
        del model
        del trainer
        torch.cuda.empty_cache()
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Benchmark completed: {config_name}")
        logger.info(f"Peak Memory: {metrics.peak_memory_gb:.2f}GB")
        logger.info(f"Training Time: {metrics.total_training_time/60:.2f} minutes")
        logger.info(f"Throughput: {metrics.samples_per_second:.2f} samples/s")
        logger.info(f"Final Eval Loss: {metrics.final_eval_loss:.4f}")
        logger.info(f"{'='*100}\n")
        
        return metrics


def main():
    """主函数：运行所有benchmark配置"""
    
    # 配置路径
    model_path = '/hpc2hdd/home/yuxuanzhao/init_model/Qwen2.5-1.5B-Instruct/'
    train_data_path = './assets/train.jsonl'
    val_data_path = './assets/validation.jsonl'
    
    # 如果文件不存在，使用当前目录下的示例数据
    if not os.path.exists(model_path):
        logger.warning(f"Model path {model_path} not found. Please update the path.")
        return
    
    if not os.path.exists(train_data_path):
        logger.warning(f"Data path {train_data_path} not found. Please update the path.")
        return
    
    # 创建benchmark实例
    benchmark = TrainingBenchmark(
        model_path=model_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path if os.path.exists(val_data_path) else None,
        output_base_dir="./benchmark_output",
        max_samples=500,  # 限制样本数以加快benchmark
        num_epochs=1  # 只训练1个epoch用于对比
    )
    
    # 定义要测试的配置
    configs = [
        # 1. 全精度微调 (需要大量显存，可能需要注释掉)
        {
            'config_name': '1_full_precision_bf16',
            'train_type': 'full',
            'quantization': None,
            'learning_rate': 1e-5,  # 全精度训练使用更小的学习率
            'batch_size': 1,  # 全精度需要更小的batch size
            'gradient_accumulation_steps': 16,
        },
        
        # 2. 标准LoRA (bf16)
        {
            'config_name': '2_lora_bf16',
            'train_type': 'lora',
            'quantization': None,
            'lora_rank': 8,
            'learning_rate': 1e-4,
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
        },
        
        # 3. QLoRA 4-bit
        {
            'config_name': '3_qlora_4bit',
            'train_type': 'lora',
            'quantization': '4bit',
            'lora_rank': 8,
            'learning_rate': 1e-4,
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
        },
        
        # 4. QLoRA 8-bit
        {
            'config_name': '4_qlora_8bit',
            'train_type': 'lora',
            'quantization': '8bit',
            'lora_rank': 8,
            'learning_rate': 1e-4,
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
        },
        
        # # 5. 更大的LoRA rank对比
        # {
        #     'config_name': '5_lora_bf16_rank16',
        #     'train_type': 'lora',
        #     'quantization': None,
        #     'lora_rank': 16,
        #     'learning_rate': 1e-4,
        #     'batch_size': 2,
        #     'gradient_accumulation_steps': 8,
        # },
    ]
    
    # 运行所有配置
    print("\n" + "="*100)
    print("STARTING COMPREHENSIVE BENCHMARK")
    print(f"Total configurations to test: {len(configs)}")
    print("="*100 + "\n")
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running configuration: {config['config_name']}")
        try:
            benchmark.run_benchmark(**config)
        except Exception as e:
            logger.error(f"Error running config {config['config_name']}: {e}")
            continue
    
    # 生成对比报告
    print("\n" + "="*100)
    print("GENERATING COMPARISON REPORTS")
    print("="*100 + "\n")
    
    benchmark.benchmark_results.save_comparison_table()
    benchmark.benchmark_results.generate_summary_report()
    
    print("\n" + "="*100)
    print("BENCHMARK COMPLETED!")
    print(f"Results saved to: {benchmark.output_base_dir}/results/")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
