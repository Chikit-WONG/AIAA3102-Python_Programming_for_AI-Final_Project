"""
Benchmark结果可视化
生成内存使用、训练时间、吞吐量等对比图表
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


@dataclass
class BenchmarkData:
    """Benchmark数据类"""
    config_name: str
    train_type: str
    quantization: str
    peak_memory_gb: float
    avg_memory_gb: float
    total_training_time: float
    samples_per_second: float
    final_eval_loss: float
    lora_rank: int


class BenchmarkVisualizer:
    """Benchmark可视化器"""
    
    def __init__(self, results_dir: str = "./benchmark_output/results"):
        self.results_dir = results_dir
        self.data: List[BenchmarkData] = []
        self.load_results()
    
    def load_results(self):
        """加载所有benchmark结果"""
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        # 排除comparison和summary文件
        json_files = [f for f in json_files if 'comparison' not in f and 'summary' not in f]
        
        for filepath in json_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                self.data.append(BenchmarkData(
                    config_name=data['config_name'],
                    train_type=data['train_type'],
                    quantization=data['quantization'] if data['quantization'] else 'None',
                    peak_memory_gb=data['peak_memory_gb'],
                    avg_memory_gb=data['avg_memory_gb'],
                    total_training_time=data['total_training_time'],
                    samples_per_second=data['samples_per_second'],
                    final_eval_loss=data['final_eval_loss'],
                    lora_rank=data.get('lora_rank', 0) or 0
                ))
        
        # 按配置名称排序
        self.data.sort(key=lambda x: x.config_name)
        
        print(f"Loaded {len(self.data)} benchmark results")
    
    def plot_memory_comparison(self, output_path: str = None):
        """绘制内存使用对比图"""
        if not self.data:
            print("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        configs = [d.config_name for d in self.data]
        peak_memory = [d.peak_memory_gb for d in self.data]
        avg_memory = [d.avg_memory_gb for d in self.data]
        
        # 峰值内存对比
        colors = ['#e74c3c' if 'full' in c.lower() else 
                  '#3498db' if 'qlora_4bit' in c.lower() else 
                  '#2ecc71' if 'qlora_8bit' in c.lower() else 
                  '#f39c12' for c in configs]
        
        bars1 = ax1.bar(range(len(configs)), peak_memory, color=colors, alpha=0.8)
        ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Peak Memory (GB)', fontsize=12, fontweight='bold')
        ax1.set_title('Peak GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, peak_memory):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}GB',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 平均内存对比
        bars2 = ax2.bar(range(len(configs)), avg_memory, color=colors, alpha=0.8)
        ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Memory (GB)', fontsize=12, fontweight='bold')
        ax2.set_title('Average GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars2, avg_memory):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}GB',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Memory comparison plot saved to: {output_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_time_comparison(self, output_path: str = None):
        """绘制训练时间对比图"""
        if not self.data:
            print("No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = [d.config_name for d in self.data]
        training_time_min = [d.total_training_time / 60 for d in self.data]
        
        colors = ['#e74c3c' if 'full' in c.lower() else 
                  '#3498db' if 'qlora_4bit' in c.lower() else 
                  '#2ecc71' if 'qlora_8bit' in c.lower() else 
                  '#f39c12' for c in configs]
        
        bars = ax.bar(range(len(configs)), training_time_min, color=colors, alpha=0.8)
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, training_time_min):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}min',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Time comparison plot saved to: {output_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_throughput_comparison(self, output_path: str = None):
        """绘制吞吐量对比图"""
        if not self.data:
            print("No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = [d.config_name for d in self.data]
        throughput = [d.samples_per_second for d in self.data]
        
        colors = ['#e74c3c' if 'full' in c.lower() else 
                  '#3498db' if 'qlora_4bit' in c.lower() else 
                  '#2ecc71' if 'qlora_8bit' in c.lower() else 
                  '#f39c12' for c in configs]
        
        bars = ax.bar(range(len(configs)), throughput, color=colors, alpha=0.8)
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (samples/second)', fontsize=12, fontweight='bold')
        ax.set_title('Training Throughput Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, throughput):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Throughput comparison plot saved to: {output_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'throughput_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_efficiency_analysis(self, output_path: str = None):
        """绘制效率分析图（内存 vs 时间）"""
        if not self.data:
            print("No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 准备数据
        peak_memory = [d.peak_memory_gb for d in self.data]
        training_time_min = [d.total_training_time / 60 for d in self.data]
        configs = [d.config_name for d in self.data]
        
        # 根据配置类型设置颜色和标记
        colors = []
        markers = []
        for d in self.data:
            if 'full' in d.config_name.lower():
                colors.append('#e74c3c')
                markers.append('s')
            elif 'qlora_4bit' in d.config_name.lower():
                colors.append('#3498db')
                markers.append('o')
            elif 'qlora_8bit' in d.config_name.lower():
                colors.append('#2ecc71')
                markers.append('^')
            else:
                colors.append('#f39c12')
                markers.append('D')
        
        # 绘制散点图
        for i, (mem, time, config, color, marker) in enumerate(zip(peak_memory, training_time_min, configs, colors, markers)):
            ax.scatter(mem, time, c=color, marker=marker, s=200, alpha=0.7, edgecolors='black', linewidths=1.5)
            ax.annotate(config.replace('_', '\n'), (mem, time), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Peak Memory (GB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_title('Training Efficiency: Memory vs Time Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Full Precision'),
            Patch(facecolor='#3498db', label='QLoRA 4-bit'),
            Patch(facecolor='#2ecc71', label='QLoRA 8-bit'),
            Patch(facecolor='#f39c12', label='LoRA (bf16)')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Efficiency analysis plot saved to: {output_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_comprehensive_comparison(self, output_path: str = None):
        """绘制综合对比图（4个子图）"""
        if not self.data:
            print("No data to plot")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        configs = [d.config_name for d in self.data]
        peak_memory = [d.peak_memory_gb for d in self.data]
        avg_memory = [d.avg_memory_gb for d in self.data]
        training_time_min = [d.total_training_time / 60 for d in self.data]
        throughput = [d.samples_per_second for d in self.data]
        
        colors = ['#e74c3c' if 'full' in c.lower() else 
                  '#3498db' if 'qlora_4bit' in c.lower() else 
                  '#2ecc71' if 'qlora_8bit' in c.lower() else 
                  '#f39c12' for c in configs]
        
        # 子图1: 峰值内存
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(configs)), peak_memory, color=colors, alpha=0.8)
        ax1.set_ylabel('Peak Memory (GB)', fontsize=11, fontweight='bold')
        ax1.set_title('Peak GPU Memory Usage', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars1, peak_memory):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 子图2: 训练时间
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(configs)), training_time_min, color=colors, alpha=0.8)
        ax2.set_ylabel('Training Time (min)', fontsize=11, fontweight='bold')
        ax2.set_title('Total Training Time', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars2, training_time_min):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 子图3: 吞吐量
        ax3 = fig.add_subplot(gs[1, 0])
        bars3 = ax3.bar(range(len(configs)), throughput, color=colors, alpha=0.8)
        ax3.set_ylabel('Throughput (samples/s)', fontsize=11, fontweight='bold')
        ax3.set_title('Training Throughput', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        for bar, value in zip(bars3, throughput):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 子图4: 效率散点图
        ax4 = fig.add_subplot(gs[1, 1])
        for i, (mem, time, config, color) in enumerate(zip(peak_memory, training_time_min, configs, colors)):
            marker = 'o' if '4bit' in config else ('^' if '8bit' in config else ('s' if 'full' in config else 'D'))
            ax4.scatter(mem, time, c=color, marker=marker, s=150, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax4.set_xlabel('Peak Memory (GB)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Training Time (min)', fontsize=11, fontweight='bold')
        ax4.set_title('Efficiency Trade-off', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加总标题
        fig.suptitle('Comprehensive Training Configuration Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive comparison plot saved to: {output_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("\nGenerating visualization plots...")
        
        self.plot_memory_comparison()
        print("✓ Memory comparison plot generated")
        
        self.plot_time_comparison()
        print("✓ Time comparison plot generated")
        
        self.plot_throughput_comparison()
        print("✓ Throughput comparison plot generated")
        
        self.plot_efficiency_analysis()
        print("✓ Efficiency analysis plot generated")
        
        self.plot_comprehensive_comparison()
        print("✓ Comprehensive comparison plot generated")
        
        print(f"\nAll plots saved to: {self.results_dir}/")


def main():
    """主函数"""
    results_dir = "./benchmark_output/results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        print("Please run the benchmark first using run_benchmark.py")
        return
    
    visualizer = BenchmarkVisualizer(results_dir=results_dir)
    
    if not visualizer.data:
        print("No benchmark data found!")
        return
    
    print(f"Found {len(visualizer.data)} benchmark results")
    print("\nConfigurations:")
    for data in visualizer.data:
        print(f"  - {data.config_name}")
    
    visualizer.generate_all_plots()
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
