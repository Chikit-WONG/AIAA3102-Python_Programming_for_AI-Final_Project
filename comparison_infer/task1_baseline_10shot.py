"""
任务1基线推理脚本（10-shot）：客户需求分类
使用原始模型（未经微调）+ 10个示例进行分类
"""

import os
import json
import random
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from swift.llm import PtEngine, InferRequest, RequestConfig, get_template
from tqdm import tqdm

# ==================== 配置参数 ====================
# 模型配置 - 使用原始模型，不加载checkpoint
model_id_or_path = '/hpc2hdd/home/yuxuanzhao/init_model/Qwen2.5-1.5B-Instruct/'
system = 'You are a helpful assistant specialized in classifying user requests.'

# 数据路径
test_data_path = './assets/test.jsonl'
validation_data_path = './assets/validation.jsonl'

# Few-shot配置
# 注意：使用平衡抽样策略，从每个类别中各选择1个样本（共11个类别）
# num_shots参数已废弃，实际示例数量由类别数量决定
random_seed = 42

# 推理配置
max_new_tokens = 128
temperature = 0
stream = False

# 输出配置
output_file = './output/task1_baseline_10shot_predictions.jsonl'  # 平衡抽样版本
save_batch_size = 10

print("="*80)
print("任务1基线推理（平衡Few-shot）：客户需求分类")
print("="*80)
print(f"模型路径: {model_id_or_path}")
print(f"方法: 平衡Few-shot (每个类别1个示例)")
print(f"Random seed: {random_seed}")
print(f"测试数据: {test_data_path}")
print(f"验证数据: {validation_data_path}")
print(f"输出文件: {output_file}")
print("="*80)

# ==================== 加载数据 ====================
def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

print(f"\n加载数据...")
test_data = load_jsonl(test_data_path)
validation_data = load_jsonl(validation_data_path)
print(f"✓ 测试样本数: {len(test_data)}")
print(f"✓ 验证样本数: {len(validation_data)}")

# ==================== 选择Few-shot示例（平衡抽样） ====================
print(f"\n从验证集选择示例（平衡抽样，random seed={random_seed}）...")
print("策略：从每个类别中各选择1个样本，确保类别覆盖")

# 按类别分组
from collections import defaultdict
category_samples = defaultdict(list)
for sample in validation_data:
    category = sample.get('category', 'UNKNOWN')
    category_samples[category].append(sample)

print(f"\n验证集类别分布:")
for cat in sorted(category_samples.keys()):
    print(f"  {cat}: {len(category_samples[cat])} 个样本")

# 从每个类别中随机选择1个样本
random.seed(random_seed)
few_shot_examples = []
categories_with_samples = sorted(category_samples.keys())

for category in categories_with_samples:
    if category_samples[category]:
        # 从该类别中随机选择1个
        sample = random.choice(category_samples[category])
        few_shot_examples.append(sample)

print(f"\n✓ 已从 {len(few_shot_examples)} 个类别中各选择1个示例")
print(f"总计: {len(few_shot_examples)} 个示例")

# 显示选中的示例
print(f"\nFew-shot 示例（按类别排序）:")
print("-" * 80)
for i, example in enumerate(few_shot_examples):
    print(f"{i+1}. Category: {example.get('category', 'N/A')}")
    print(f"   Instruction: {example['instruction'][:80]}...")
    print()

# ==================== 初始化模型 ====================
print(f"初始化模型（原始模型，无微调）...")
engine = PtEngine(model_id_or_path)
template = get_template(engine.model_meta.template, engine.processor, default_system=system)
engine.default_template = template
print(f"✓ 模型加载完成")

# ==================== 构建Few-shot Prompt ====================
def build_few_shot_prompt(instruction, examples):
    """
    构建包含few-shot示例的prompt
    
    Args:
        instruction: 当前要分类的用户请求
        examples: few-shot示例列表
    
    Returns:
        完整的prompt
    """
    prompt_parts = []
    
    # 添加任务说明和类别列表
    prompt_parts.append("Task: Classify user requests into one of these categories:")
    prompt_parts.append("FEEDBACK, REFUND, ACCOUNT, ORDER, SHIPPING, SUBSCRIPTION, PAYMENT, CANCEL, DELIVERY, INVOICE, CONTACT")
    prompt_parts.append("")
    prompt_parts.append("Here are some examples:")
    prompt_parts.append("")
    
    # 添加示例
    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"User request: {example['instruction']}")
        prompt_parts.append(f"Category: {example['category']}")
        prompt_parts.append("")
    
    # 添加当前任务
    prompt_parts.append("Now, classify this user request:")
    prompt_parts.append(f"User request: {instruction}")
    prompt_parts.append("Category:")
    
    return "\n".join(prompt_parts)

# ==================== 推理函数 ====================
def predict_category(engine, instruction, examples):
    """
    对单个样本进行分类预测（10-shot）
    
    Args:
        engine: 推理引擎
        instruction: 用户请求文本
        examples: few-shot示例
    
    Returns:
        predicted_category: 预测的类别
    """
    # 构造包含示例的prompt
    query = build_few_shot_prompt(instruction, examples)
    
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
        predicted_category = predict_category(engine, instruction, few_shot_examples)
        
        # 保存结果
        result = {
            'instruction': instruction,
            'predicted_category': predicted_category,
            'ground_truth_category': item.get('category', None),
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