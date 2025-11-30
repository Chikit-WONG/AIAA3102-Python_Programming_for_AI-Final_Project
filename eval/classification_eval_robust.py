import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import os
import re
warnings.filterwarnings('ignore')

# 定义有效的类别列表
VALID_CATEGORIES = ['FEEDBACK', 'REFUND', 'ACCOUNT', 'ORDER', 'SHIPPING', 
                    'SUBSCRIPTION', 'PAYMENT', 'CANCEL', 'DELIVERY', 'INVOICE', 'CONTACT']

def extract_category_from_text(text):
    """
    从文本中提取第一个出现的有效类别词（按文本顺序，不是按列表顺序）
    
    Args:
        text: 模型输出的文本（可能包含解释性内容）
    
    Returns:
        提取的类别，如果没有找到则返回"EXTRACTION_FAILED"
    """
    if not isinstance(text, str):
        return "EXTRACTION_FAILED"
    
    # 转换为大写以进行不区分大小写的匹配
    text_upper = text.upper()
    
    # 方法1: 寻找第一个完整匹配的类别词（按在文本中出现的位置排序）
    matches = []
    for category in VALID_CATEGORIES:
        # 使用正则表达式匹配独立的单词（不是其他单词的一部分）
        pattern = r'\b' + re.escape(category) + r'\b'
        match = re.search(pattern, text_upper)
        if match:
            matches.append((match.start(), category))
    
    # 如果找到完整匹配，返回第一个出现的
    if matches:
        matches.sort(key=lambda x: x[0])  # 按位置排序
        return matches[0][1]
    
    # 方法2: 如果没有找到独立单词，寻找第一个包含的类别词（按在文本中出现的位置）
    matches = []
    for category in VALID_CATEGORIES:
        pos = text_upper.find(category)
        if pos != -1:
            matches.append((pos, category))
    
    if matches:
        matches.sort(key=lambda x: x[0])  # 按位置排序
        return matches[0][1]
    
    # 如果都没找到，返回特殊标签表示提取失败
    return "EXTRACTION_FAILED"

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_categories(data):
    """从数据中提取预测类别和真实类别（支持从冗长回答中提取）"""
    predicted_raw = [item['predicted_category'] for item in data]
    ground_truth = [item['ground_truth_category'] for item in data]
    
    # 从预测结果中提取实际类别
    predicted = [extract_category_from_text(pred) for pred in predicted_raw]
    
    # 打印一些提取示例以便验证
    print("\n" + "="*60)
    print("类别提取示例（前5个）")
    print("="*60)
    extraction_failed = 0
    for i in range(min(5, len(predicted_raw))):
        print(f"\n样本 {i+1}:")
        print(f"  原始输出: {predicted_raw[i][:100]}...")
        print(f"  提取类别: {predicted[i]}")
        print(f"  真实类别: {ground_truth[i]}")
        if predicted[i] == "EXTRACTION_FAILED":
            extraction_failed += 1
            print(f"  ⚠️  警告: 提取失败，将被标记为EXTRACTION_FAILED")
    
    # 统计提取失败的数量
    total_failed = sum(1 for p in predicted if p == "EXTRACTION_FAILED")
    if total_failed > 0:
        print(f"\n⚠️  总计 {total_failed}/{len(predicted)} 个样本提取失败")
        print(f"这些样本将在评估中被标记为'EXTRACTION_FAILED'类别")
        print(f"提取失败率: {total_failed/len(predicted)*100:.2f}%")
    else:
        print(f"\n✓ 所有 {len(predicted)} 个样本都成功提取了有效类别")
    
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

def save_extraction_report(data, predicted, ground_truth, save_path='extraction_report.txt'):
    """保存提取报告，记录所有提取失败的样本"""
    failed_extractions = []
    
    for i, (item, pred, gt) in enumerate(zip(data, predicted, ground_truth)):
        if pred == "EXTRACTION_FAILED":
            failed_extractions.append({
                'index': i,
                'original_output': item['predicted_category'],
                'extracted': pred,
                'ground_truth': gt
            })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("类别提取报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"总样本数: {len(data)}\n")
        f.write(f"成功提取: {len(predicted) - len(failed_extractions)}\n")
        f.write(f"提取失败: {len(failed_extractions)}\n")
        f.write(f"成功率: {(len(predicted) - len(failed_extractions)) / len(predicted) * 100:.2f}%\n\n")
        
        if failed_extractions:
            f.write("="*80 + "\n")
            f.write("提取失败的样本详情\n")
            f.write("="*80 + "\n\n")
            f.write("注意: 这些样本在评估中被标记为'EXTRACTION_FAILED'类别\n")
            f.write("这表示模型输出中未包含任何预定义的有效类别词\n\n")
            
            for fail in failed_extractions:
                f.write(f"样本 {fail['index'] + 1}:\n")
                f.write(f"原始输出: {fail['original_output']}\n")
                f.write(f"提取结果: {fail['extracted']}\n")
                f.write(f"真实类别: {fail['ground_truth']}\n")
                f.write("-" * 80 + "\n\n")
        else:
            f.write("\n✓ 所有样本都成功提取了有效类别！\n")
    
    print(f"提取报告已保存到: {save_path}")

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
    
    # 提取类别（支持从冗长回答中提取）
    predicted, ground_truth = extract_categories(data)
    
    # 保存提取报告
    extraction_report_path = os.path.join(output_dir, 'extraction_report.txt')
    save_extraction_report(data, predicted, ground_truth, save_path=extraction_report_path)
    
    # 获取所有唯一的类别标签(排序后)
    # 优先使用预定义的有效类别列表
    labels_set = set(ground_truth + predicted)
    labels = [cat for cat in VALID_CATEGORIES if cat in labels_set]
    
    # 添加EXTRACTION_FAILED标签（如果存在）
    if "EXTRACTION_FAILED" in labels_set:
        labels.append("EXTRACTION_FAILED")
    
    # 添加其他不在预定义列表中的类别（如果有，理论上不应该出现）
    other_labels = sorted([cat for cat in labels_set 
                          if cat not in VALID_CATEGORIES and cat != "EXTRACTION_FAILED"])
    if other_labels:
        labels.extend(other_labels)
    
    print(f"\n发现 {len(labels)} 个类别: {labels}")
    if "EXTRACTION_FAILED" in labels:
        failed_count = sum(1 for p in predicted if p == "EXTRACTION_FAILED")
        print(f"⚠️  其中包含 {failed_count} 个EXTRACTION_FAILED（提取失败）样本")
    if other_labels:
        print(f"⚠️  警告: 发现 {len(other_labels)} 个意外类别: {other_labels}")
    
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
    print(f"  5. extraction_report.txt - 类别提取报告")

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        jsonl_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # 默认路径(修改为你的文件路径)
        jsonl_file = "./output/task1_baseline_10shot_predictions.jsonl"
        output_dir = "./eval_output/classification_10shot"  # 使用输入文件所在目录
    
    try:
        main(jsonl_file, output_dir)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()