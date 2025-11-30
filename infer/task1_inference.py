"""
任务1推理脚本：客户需求分类
读取 test.jsonl，对每个样本进行分类预测
"""

import os
import json
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from swift.llm import PtEngine, InferRequest, RequestConfig, get_template
from tqdm import tqdm

# ==================== 配置参数 ====================
# 模型配置
model_id_or_path = '/hpc2hdd/home/yuxuanzhao/init_model/Qwen2.5-1.5B-Instruct/'
checkpoint_path = 'checkpoint/task1_classification/final_model'  # 或使用特定的checkpoint
system = 'You are a helpful assistant specialized in classifying user requests.'

# 测试数据
test_data_path = './assets/test.jsonl'

# 推理配置
max_new_tokens = 128  # 分类任务只需要短输出
temperature = 0  # 使用确定性输出
stream = False  # 分类任务不需要流式输出

# 输出配置
output_file = './output/task1_predictions.jsonl'
save_batch_size = 10  # 每处理10个样本保存一次

print("="*80)
print("任务1推理：客户需求分类")
print("="*80)
print(f"模型路径: {model_id_or_path}")
print(f"检查点路径: {checkpoint_path}")
print(f"测试数据: {test_data_path}")
print(f"输出文件: {output_file}")
print("="*80)

# ==================== 加载测试数据 ====================
def load_test_data(file_path):
    """加载测试数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

print(f"\n加载测试数据...")
test_data = load_test_data(test_data_path)
print(f"✓ 测试样本数: {len(test_data)}")

# ==================== 初始化模型 ====================
print(f"\n初始化模型...")
engine = PtEngine(model_id_or_path, adapters=[checkpoint_path])
template = get_template(engine.model_meta.template, engine.processor, default_system=system)
engine.default_template = template
print(f"✓ 模型加载完成")

# ==================== 推理函数 ====================
def predict_category(engine, instruction):
    """
    对单个样本进行分类预测
    
    Args:
        engine: 推理引擎
        instruction: 用户请求文本
    
    Returns:
        predicted_category: 预测的类别
    """
    # 构造输入（与训练时保持一致）
    query = f"Please classify the following user request into the appropriate category: {instruction}"
    
    # 创建推理请求
    infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=False)
    
    # 执行推理
    resp_list = engine.infer([infer_request], request_config)
    predicted_category = resp_list[0].choices[0].message.content.strip()
    
    return predicted_category

# ==================== 批量推理 ====================
print(f"\n开始推理...")
print(f"提示：推理过程中会每处理 {save_batch_size} 个样本保存一次结果")
print("-"*80)

results = []
start_time = datetime.now()

for idx, item in enumerate(tqdm(test_data, desc="推理进度")):
    try:
        # 获取instruction
        instruction = item['instruction']
        
        # 进行预测
        predicted_category = predict_category(engine, instruction)
        
        # 保存结果
        result = {
            'instruction': instruction,
            'predicted_category': predicted_category,
            'ground_truth_category': item.get('category', None),  # 如果test.jsonl有标签
        }
        
        # 如果test.jsonl包含其他字段，也保留
        if 'response' in item:
            result['ground_truth_response'] = item['response']
        
        results.append(result)
        
        # 定期保存结果
        if (idx + 1) % save_batch_size == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
    except Exception as e:
        print(f"\n错误：处理样本 {idx} 时出错: {e}")
        print(f"样本内容: {item}")
        continue

# ==================== 保存最终结果 ====================
print(f"\n保存最终结果...")
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

end_time = datetime.now()
duration = end_time - start_time

print(f"\n{'='*80}")
print(f"推理完成！")
print(f"{'='*80}")
print(f"总样本数: {len(test_data)}")
print(f"成功处理: {len(results)}")
print(f"失败数量: {len(test_data) - len(results)}")
print(f"总耗时: {duration}")
print(f"平均速度: {len(results) / duration.total_seconds():.2f} 样本/秒")
print(f"结果已保存到: {output_file}")
print(f"{'='*80}")

# ==================== 计算准确率（如果有标签） ====================
if test_data and 'category' in test_data[0]:
    print(f"\n计算分类准确率...")
    correct = 0
    total = 0
    
    for result in results:
        if result['ground_truth_category'] is not None:
            total += 1
            if result['predicted_category'] == result['ground_truth_category']:
                correct += 1
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"准确率: {correct}/{total} = {accuracy:.2f}%")
        
        # 统计每个类别的准确率
        from collections import defaultdict
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in results:
            if result['ground_truth_category'] is not None:
                cat = result['ground_truth_category']
                category_stats[cat]['total'] += 1
                if result['predicted_category'] == result['ground_truth_category']:
                    category_stats[cat]['correct'] += 1
        
        print(f"\n各类别准确率:")
        print(f"{'类别':<20} {'正确/总数':<15} {'准确率':<10}")
        print("-" * 50)
        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            cat_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{cat:<20} {stats['correct']}/{stats['total']:<10} {cat_acc:>6.2f}%")

# ==================== 显示预测示例 ====================
print(f"\n预测示例（前5个）:")
print("-" * 80)
for i, result in enumerate(results[:5]):
    print(f"\n样本 {i+1}:")
    print(f"  Instruction: {result['instruction'][:100]}...")
    print(f"  预测类别: {result['predicted_category']}")
    if result['ground_truth_category']:
        match = "✓" if result['predicted_category'] == result['ground_truth_category'] else "✗"
        print(f"  真实类别: {result['ground_truth_category']} {match}")

print(f"\n{'='*80}")
print(f"推理完成！完整结果已保存到: {output_file}")
print(f"{'='*80}\n")
