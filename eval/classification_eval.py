import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import os
warnings.filterwarnings('ignore')

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_categories(data):
    """从数据中提取预测类别和真实类别"""
    predicted = [item['predicted_category'] for item in data]
    ground_truth = [item['ground_truth_category'] for item in data]
    return predicted, ground_truth

def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {save_path}")
    plt.close()
    
    return cm

def plot_normalized_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix_normalized.png'):
    """绘制归一化混淆矩阵(按行归一化,显示百分比)"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 按行归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除以0的情况
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Percentage'})
    
    plt.title('Normalized Confusion Matrix (by Row)', fontsize=16, pad=20)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存到: {save_path}")
    plt.close()

def calculate_metrics(y_true, y_pred, labels):
    """计算各种分类指标"""
    # 整体准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 宏平均和微平均指标
    precision_macro = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, labels=labels, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, labels=labels, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    
    # 打印整体指标
    print("\n" + "="*60)
    print("整体分类指标")
    print("="*60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"\n宏平均 (Macro Average):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    print(f"\n微平均 (Micro Average):")
    print(f"  Precision: {precision_micro:.4f}")
    print(f"  Recall:    {recall_micro:.4f}")
    print(f"  F1-Score:  {f1_micro:.4f}")
    print(f"\n加权平均 (Weighted Average):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall:    {recall_weighted:.4f}")
    print(f"  F1-Score:  {f1_weighted:.4f}")
    
    # 每个类别的详细报告
    print("\n" + "="*60)
    print("每个类别的详细指标")
    print("="*60)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    
    # 保存详细报告到文件
    report_dict = classification_report(y_true, y_pred, labels=labels, 
                                       output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'detailed_report': report_dict
    }

def save_metrics_to_csv(metrics, labels, save_path='classification_metrics.csv'):
    """将指标保存为CSV文件"""
    # 提取每个类别的指标
    rows = []
    for label in labels:
        if label in metrics['detailed_report']:
            row = {
                'Category': label,
                'Precision': metrics['detailed_report'][label]['precision'],
                'Recall': metrics['detailed_report'][label]['recall'],
                'F1-Score': metrics['detailed_report'][label]['f1-score'],
                'Support': metrics['detailed_report'][label]['support']
            }
            rows.append(row)
    
    # 添加平均指标
    rows.append({
        'Category': 'Macro Avg',
        'Precision': metrics['precision_macro'],
        'Recall': metrics['recall_macro'],
        'F1-Score': metrics['f1_macro'],
        'Support': '-'
    })
    
    rows.append({
        'Category': 'Weighted Avg',
        'Precision': metrics['precision_weighted'],
        'Recall': metrics['recall_weighted'],
        'F1-Score': metrics['f1_weighted'],
        'Support': '-'
    })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\n指标已保存到CSV文件: {save_path}")

def plot_metrics_comparison(metrics, labels, save_path='metrics_comparison.png'):
    """绘制各类别指标对比图"""
    # 提取数据
    categories = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for label in labels:
        if label in metrics['detailed_report']:
            categories.append(label)
            precisions.append(metrics['detailed_report'][label]['precision'])
            recalls.append(metrics['detailed_report'][label]['recall'])
            f1_scores.append(metrics['detailed_report'][label]['f1-score'])
    
    # 创建图形
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison by Category', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"指标对比图已保存到: {save_path}")
    plt.close()

def main(jsonl_file_path, output_dir=None):
    """主函数
    
    Args:
        jsonl_file_path: JSONL文件路径
        output_dir: 输出目录路径,如果为None则使用输入文件所在目录
    """
    print("="*60)
    print("开始分析JSONL文件")
    print("="*60)
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(jsonl_file_path))
    
    # 创建输出目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 读取数据
    print(f"\n正在读取文件: {jsonl_file_path}")
    try:
        data = read_jsonl(jsonl_file_path)
        print(f"成功读取 {len(data)} 条数据")
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{jsonl_file_path}'")
        print("请确保文件路径正确且文件存在")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
    
    # 提取类别
    predicted, ground_truth = extract_categories(data)
    
    # 获取所有唯一的类别标签(排序后)
    labels = sorted(list(set(ground_truth + predicted)))
    print(f"\n发现 {len(labels)} 个类别: {labels}")
    
    # 定义输出文件路径
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    metrics_csv_path = os.path.join(output_dir, 'classification_metrics.csv')
    comparison_path = os.path.join(output_dir, 'metrics_comparison.png')
    
    # 绘制混淆矩阵
    print("\n正在生成混淆矩阵...")
    try:
        cm = plot_confusion_matrix(ground_truth, predicted, labels, save_path=cm_path)
    except Exception as e:
        print(f"生成混淆矩阵时发生错误: {e}")
        return
    
    # 绘制归一化混淆矩阵
    print("正在生成归一化混淆矩阵...")
    try:
        plot_normalized_confusion_matrix(ground_truth, predicted, labels, save_path=cm_norm_path)
    except Exception as e:
        print(f"生成归一化混淆矩阵时发生错误: {e}")
    
    # 计算指标
    print("\n正在计算分类指标...")
    try:
        metrics = calculate_metrics(ground_truth, predicted, labels)
    except Exception as e:
        print(f"计算指标时发生错误: {e}")
        return
    
    # 保存指标到CSV
    try:
        save_metrics_to_csv(metrics, labels, save_path=metrics_csv_path)
    except Exception as e:
        print(f"保存CSV文件时发生错误: {e}")
    
    # 绘制指标对比图
    print("\n正在生成指标对比图...")
    try:
        plot_metrics_comparison(metrics, labels, save_path=comparison_path)
    except Exception as e:
        print(f"生成指标对比图时发生错误: {e}")
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    print(f"\n所有文件已保存到: {output_dir}")
    print("\n生成的文件:")
    print(f"  1. confusion_matrix.png - 混淆矩阵")
    print(f"  2. confusion_matrix_normalized.png - 归一化混淆矩阵")
    print(f"  3. classification_metrics.csv - 分类指标表")
    print(f"  4. metrics_comparison.png - 指标对比图")

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        jsonl_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # 默认路径(修改为你的文件路径)
        jsonl_file = "/hpc2hdd/home/yuxuanzhao/haodong/3102project/task1_predictions.jsonl"
        output_dir = "./output/classification"  # 使用输入文件所在目录
    
    try:
        main(jsonl_file, output_dir)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()