"""
性能基准测试工具模块
提供GPU内存监控、训练时间记录等功能
"""

import torch
import time
import psutil
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np


@dataclass
class MemorySnapshot:
    """GPU内存快照"""
    timestamp: float
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    max_reserved_gb: float
    free_gb: float
    total_gb: float


@dataclass
class TrainingMetrics:
    """训练指标"""
    config_name: str
    model_name: str
    train_type: str  # 'full', 'lora', 'qlora_4bit', 'qlora_8bit'
    quantization: Optional[str]  # None, '4bit', '8bit'
    
    # 时间指标
    total_training_time: float
    avg_step_time: float
    samples_per_second: float
    
    # 内存指标
    peak_memory_gb: float
    avg_memory_gb: float
    memory_snapshots: List[Dict]
    
    # 训练指标
    final_train_loss: float
    final_eval_loss: float
    total_steps: int
    
    # 系统信息
    gpu_name: str
    gpu_total_memory_gb: float
    torch_version: str
    cuda_version: str
    
    # 配置信息
    batch_size: int
    gradient_accumulation_steps: int
    lora_rank: Optional[int]
    learning_rate: float
    
    timestamp: str


class GPUMemoryMonitor:
    """GPU内存监控器"""
    
    def __init__(self, device: int = 0):
        self.device = device
        self.snapshots: List[MemorySnapshot] = []
        self.is_available = torch.cuda.is_available()
        
        if self.is_available:
            self.gpu_name = torch.cuda.get_device_name(device)
            self.gpu_total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            self.gpu_name = "N/A"
            self.gpu_total_memory = 0
    
    def reset(self):
        """重置监控器"""
        self.snapshots = []
        if self.is_available:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
    
    def capture_snapshot(self) -> Optional[MemorySnapshot]:
        """捕获当前内存快照"""
        if not self.is_available:
            return None
        
        free_mem, total_mem = torch.cuda.mem_get_info(self.device)
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=torch.cuda.memory_allocated(self.device) / (1024**3),
            reserved_gb=torch.cuda.memory_reserved(self.device) / (1024**3),
            max_allocated_gb=torch.cuda.max_memory_allocated(self.device) / (1024**3),
            max_reserved_gb=torch.cuda.max_memory_reserved(self.device) / (1024**3),
            free_gb=free_mem / (1024**3),
            total_gb=total_mem / (1024**3)
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_peak_memory(self) -> float:
        """获取峰值内存使用（GB）"""
        if not self.snapshots:
            return 0.0
        return max(s.max_allocated_gb for s in self.snapshots)
    
    def get_avg_memory(self) -> float:
        """获取平均内存使用（GB）"""
        if not self.snapshots:
            return 0.0
        return np.mean([s.allocated_gb for s in self.snapshots])
    
    def get_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.snapshots:
            return {}
        
        allocated = [s.allocated_gb for s in self.snapshots]
        return {
            'peak_allocated_gb': max(allocated),
            'avg_allocated_gb': np.mean(allocated),
            'min_allocated_gb': min(allocated),
            'final_allocated_gb': allocated[-1],
            'gpu_name': self.gpu_name,
            'gpu_total_memory_gb': self.gpu_total_memory
        }


class BenchmarkCallback(TrainerCallback):
    """训练性能监控回调"""
    
    def __init__(self, memory_monitor: GPUMemoryMonitor, log_every_n_steps: int = 10):
        self.memory_monitor = memory_monitor
        self.log_every_n_steps = log_every_n_steps
        self.step_times: List[float] = []
        self.step_start_time: Optional[float] = None
        self.training_start_time: Optional[float] = None
        self.training_end_time: Optional[float] = None
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时的回调"""
        self.training_start_time = time.time()
        self.memory_monitor.reset()
        self.step_times = []
        print(f"\n{'='*80}")
        print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """每个训练步骤开始时的回调"""
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """每个训练步骤结束时的回调"""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
        
        # 定期捕获内存快照
        if state.global_step % self.log_every_n_steps == 0:
            snapshot = self.memory_monitor.capture_snapshot()
            if snapshot:
                print(f"Step {state.global_step}: "
                      f"Memory: {snapshot.allocated_gb:.2f}GB / {snapshot.total_gb:.2f}GB, "
                      f"Peak: {snapshot.max_allocated_gb:.2f}GB, "
                      f"Step time: {step_time:.3f}s")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时的回调"""
        self.training_end_time = time.time()
        total_time = self.training_end_time - self.training_start_time
        
        print(f"\n{'='*80}")
        print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        print(f"Average step time: {np.mean(self.step_times):.3f}s")
        print(f"Peak GPU memory: {self.memory_monitor.get_peak_memory():.2f}GB")
        print(f"Average GPU memory: {self.memory_monitor.get_avg_memory():.2f}GB")
        print(f"{'='*80}\n")
    
    def get_total_training_time(self) -> float:
        """获取总训练时间（秒）"""
        if self.training_start_time and self.training_end_time:
            return self.training_end_time - self.training_start_time
        return 0.0
    
    def get_avg_step_time(self) -> float:
        """获取平均步骤时间（秒）"""
        return np.mean(self.step_times) if self.step_times else 0.0


class BenchmarkResults:
    """基准测试结果管理"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[TrainingMetrics] = []
    
    def add_result(self, metrics: TrainingMetrics):
        """添加一个训练结果"""
        self.results.append(metrics)
        self._save_result(metrics)
    
    def _save_result(self, metrics: TrainingMetrics):
        """保存单个结果到JSON文件"""
        filename = f"{metrics.config_name}_{metrics.timestamp.replace(':', '-').replace(' ', '_')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filepath}")
    
    def save_comparison_table(self, filename: str = "comparison_table.txt"):
        """保存对比表格"""
        if not self.results:
            print("No results to compare")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write("Training Configuration Comparison\n")
            f.write("="*120 + "\n\n")
            
            # 表头
            header = f"{'Config':<20} {'Train Type':<15} {'Quant':<10} {'Peak Mem(GB)':<15} {'Avg Mem(GB)':<15} {'Time(min)':<12} {'Samples/s':<12} {'Final Loss':<12}\n"
            f.write(header)
            f.write("-"*120 + "\n")
            
            # 数据行
            for metrics in self.results:
                quant = metrics.quantization if metrics.quantization else "None"
                row = (f"{metrics.config_name:<20} "
                       f"{metrics.train_type:<15} "
                       f"{quant:<10} "
                       f"{metrics.peak_memory_gb:<15.2f} "
                       f"{metrics.avg_memory_gb:<15.2f} "
                       f"{metrics.total_training_time/60:<12.2f} "
                       f"{metrics.samples_per_second:<12.2f} "
                       f"{metrics.final_eval_loss:<12.4f}\n")
                f.write(row)
            
            f.write("="*120 + "\n")
        
        print(f"Comparison table saved to: {filepath}")
        
        # 同时打印到控制台
        with open(filepath, 'r') as f:
            print(f.read())
    
    def generate_summary_report(self, filename: str = "summary_report.txt"):
        """生成详细的摘要报告"""
        if not self.results:
            print("No results to report")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("BENCHMARK SUMMARY REPORT\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            # 按配置类型分组
            full_precision = [m for m in self.results if m.train_type == 'full']
            lora = [m for m in self.results if m.train_type == 'lora' and not m.quantization]
            qlora_4bit = [m for m in self.results if m.quantization == '4bit']
            qlora_8bit = [m for m in self.results if m.quantization == '8bit']
            
            # 计算对比基准（通常是全精度）
            baseline = full_precision[0] if full_precision else self.results[0]
            
            f.write(f"Baseline Configuration: {baseline.config_name}\n")
            f.write(f"  - Training Type: {baseline.train_type}\n")
            f.write(f"  - Peak Memory: {baseline.peak_memory_gb:.2f} GB\n")
            f.write(f"  - Training Time: {baseline.total_training_time/60:.2f} minutes\n")
            f.write(f"  - Throughput: {baseline.samples_per_second:.2f} samples/s\n\n")
            
            f.write("-"*100 + "\n\n")
            
            # 详细对比每个配置
            for metrics in self.results:
                if metrics == baseline:
                    continue
                
                mem_reduction = ((baseline.peak_memory_gb - metrics.peak_memory_gb) / baseline.peak_memory_gb) * 100
                time_change = ((metrics.total_training_time - baseline.total_training_time) / baseline.total_training_time) * 100
                throughput_change = ((metrics.samples_per_second - baseline.samples_per_second) / baseline.samples_per_second) * 100
                
                f.write(f"Configuration: {metrics.config_name}\n")
                f.write(f"  Training Type: {metrics.train_type}\n")
                f.write(f"  Quantization: {metrics.quantization if metrics.quantization else 'None'}\n")
                f.write(f"  \n")
                f.write(f"  Memory:\n")
                f.write(f"    Peak: {metrics.peak_memory_gb:.2f} GB ({mem_reduction:+.1f}% vs baseline)\n")
                f.write(f"    Average: {metrics.avg_memory_gb:.2f} GB\n")
                f.write(f"  \n")
                f.write(f"  Performance:\n")
                f.write(f"    Training Time: {metrics.total_training_time/60:.2f} min ({time_change:+.1f}% vs baseline)\n")
                f.write(f"    Avg Step Time: {metrics.avg_step_time:.3f}s\n")
                f.write(f"    Throughput: {metrics.samples_per_second:.2f} samples/s ({throughput_change:+.1f}% vs baseline)\n")
                f.write(f"  \n")
                f.write(f"  Training Quality:\n")
                f.write(f"    Final Training Loss: {metrics.final_train_loss:.4f}\n")
                f.write(f"    Final Eval Loss: {metrics.final_eval_loss:.4f}\n")
                f.write(f"  \n")
                f.write(f"  Configuration Details:\n")
                f.write(f"    Batch Size: {metrics.batch_size}\n")
                f.write(f"    Gradient Accumulation: {metrics.gradient_accumulation_steps}\n")
                f.write(f"    Learning Rate: {metrics.learning_rate}\n")
                if metrics.lora_rank:
                    f.write(f"    LoRA Rank: {metrics.lora_rank}\n")
                f.write(f"\n")
                f.write("-"*100 + "\n\n")
            
            # 关键发现总结
            f.write("KEY FINDINGS:\n\n")
            
            if qlora_4bit:
                m = qlora_4bit[0]
                mem_save = ((baseline.peak_memory_gb - m.peak_memory_gb) / baseline.peak_memory_gb) * 100
                f.write(f"1. QLoRA 4-bit reduces memory by ~{mem_save:.1f}% compared to full precision\n")
            
            if qlora_8bit:
                m = qlora_8bit[0]
                mem_save = ((baseline.peak_memory_gb - m.peak_memory_gb) / baseline.peak_memory_gb) * 100
                f.write(f"2. QLoRA 8-bit reduces memory by ~{mem_save:.1f}% compared to full precision\n")
            
            if lora:
                m = lora[0]
                mem_save = ((baseline.peak_memory_gb - m.peak_memory_gb) / baseline.peak_memory_gb) * 100
                f.write(f"3. LoRA (bf16) reduces memory by ~{mem_save:.1f}% compared to full precision\n")
            
            f.write(f"\n")
            f.write("="*100 + "\n")
        
        print(f"Summary report saved to: {filepath}")


def get_system_info() -> Dict:
    """获取系统信息"""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info
