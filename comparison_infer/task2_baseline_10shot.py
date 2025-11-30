"""
任务2基线推理脚本（10-shot）：用户回复生成
使用原始模型（未经微调）+ 10个示例进行回复生成
"""

import os
import json
import random
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import PtEngine, InferRequest, RequestConfig, get_template
from tqdm import tqdm

# ==================== 配置参数 ====================
# 模型配置 - 使用原始模型，不加载checkpoint
model_id_or_path = '/hpc2hdd/home/yuxuanzhao/init_model/Qwen2.5-1.5B-Instruct/'
system = 'You are a helpful customer service assistant. Generate appropriate responses to user requests based on their category.'

# 数据路径
test_data_path = './assets/test.jsonl'
validation_data_path = './assets/validation.jsonl'

# Few-shot配置
# 注意：使用平衡抽样策略，从每个类别中各选择1个样本（共11个类别）
# num_shots参数已废弃，实际示例数量由类别数量决定
random_seed = 42

# 推理配置
max_new_tokens = 512
temperature = 0.7
stream = False

# 输出配置
output_file = './output/task2_baseline_10shot_predictions.jsonl'  # 平衡抽样版本
save_batch_size = 10

print("="*80)
print("任务2基线推理（平衡Few-shot）：用户回复生成")
print("="*80)
print(f"模型路径: {model_id_or_path}")
print(f"方法: 平衡Few-shot (每个类别1个示例)")
print(f"Random seed: {random_seed}")
print(f"测试数据: {test_data_path}")
print(f"验证数据: {validation_data_path}")
print(f"输出文件: {output_file}")
print(f"Temperature: {temperature}")
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

# 检查数据是否包含category字段
if test_data and 'category' not in test_data[0]:
    print("\n⚠️  警告: test.jsonl 不包含 'category' 字段")
    print("建议先运行 task1_inference.py 获取分类结果，或确保test.jsonl包含category字段")
    response = input("是否继续？(y/n): ")
    if response.lower() != 'y':
        exit(0)

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
    print(f"   Instruction: {example['instruction'][:60]}...")
    print(f"   Response: {example.get('response', 'N/A')[:60]}...")
    print()

# ==================== 初始化模型 ====================
print(f"初始化模型（原始模型，无微调）...")
engine = PtEngine(model_id_or_path)
template = get_template(engine.model_meta.template, engine.processor, default_system=system)
engine.default_template = template
print(f"✓ 模型加载完成")

# ==================== 构建Few-shot Prompt ====================
def build_few_shot_prompt(instruction, category, examples):
    """
    构建包含few-shot示例的prompt
    
    Args:
        instruction: 当前用户请求
        category: 请求类别
        examples: few-shot示例列表
    
    Returns:
        完整的prompt
    """
    prompt_parts = []
    
    # 添加任务说明
    prompt_parts.append("Here are some examples of user requests and appropriate responses:")
    prompt_parts.append("")
    
    # 添加示例
    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Category: Type {example.get('category', 'UNKNOWN')}")
        prompt_parts.append(f"User request: {example['instruction']}")
        prompt_parts.append(f"Response: {example.get('response', 'N/A')}")
        prompt_parts.append("")
    
    # 添加当前任务
    prompt_parts.append("Now, please generate an appropriate response for the following user request:")
    prompt_parts.append(f"Category: Type {category}")
    prompt_parts.append(f"User request: {instruction}")
    prompt_parts.append("Response:")
    
    return "\n".join(prompt_parts)

# ==================== 推理函数 ====================
def generate_response(engine, instruction, category, examples, use_stream=False):
    """
    为单个样本生成回复（10-shot）
    
    Args:
        engine: 推理引擎
        instruction: 用户请求文本
        category: 请求类别
        examples: few-shot示例
        use_stream: 是否使用流式输出
    
    Returns:
        generated_response: 生成的回复
    """
    # 构造包含示例的prompt
    query = build_few_shot_prompt(instruction, category, examples)
    
    # 创建推理请求
    infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=use_stream)
    
    if use_stream:
        # 流式输出
        gen_list = engine.infer([infer_request], request_config)
        generated_response = ""
        for resp in gen_list[0]:
            if resp is None:
                continue
            chunk = resp.choices[0].delta.content
            generated_response += chunk
            print(chunk, end='', flush=True)
        print()
    else:
        # 非流式输出
        resp_list = engine.infer([infer_request], request_config)
        generated_response = resp_list[0].choices[0].message.content.strip()
    
    return generated_response

# ==================== 批量推理 ====================
print(f"\n开始推理...")
print(f"提示：推理过程中会每处理 {save_batch_size} 个样本保存一次结果")
if stream:
    print(f"已启用流式输出，将实时显示生成过程")
print("-"*80)

results = []
start_time = datetime.now()

for idx, item in enumerate(tqdm(test_data, desc="推理进度", disable=stream)):
    try:
        # 获取instruction和category
        instruction = item['instruction']
        category = item.get('category', 'UNKNOWN')
        
        if stream:
            print(f"\n样本 {idx+1}/{len(test_data)}:")
            print(f"Category: {category}")
            print(f"Instruction: {instruction[:100]}...")
            print(f"生成回复: ", end='', flush=True)
        
        # 生成回复
        generated_response = generate_response(engine, instruction, category, few_shot_examples, use_stream=stream)
        
        # 保存结果
        result = {
            'instruction': instruction,
            'category': category,
            'generated_response': generated_response,
            'ground_truth_response': item.get('response', None),
        }
        
        results.append(result)
        
        # 定期保存结果
        if (idx + 1) % save_batch_size == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            if not stream:
                print(f"\n已保存 {idx + 1} 个样本的结果")
        
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

# ==================== 显示生成示例 ====================
print(f"\n生成示例（前3个）:")
print("="*80)
for i, result in enumerate(results[:3]):
    print(f"\n样本 {i+1}:")
    print(f"  类别: {result['category']}")
    print(f"  请求: {result['instruction'][:80]}...")
    print(f"  生成回复: {result['generated_response'][:150]}...")
    if result['ground_truth_response']:
        print(f"  真实回复: {result['ground_truth_response'][:150]}...")
    print("-" * 80)

print(f"\n{'='*80}")
print(f"推理完成！完整结果已保存到: {output_file}")
print(f"{'='*80}\n")