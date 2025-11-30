"""
任务2推理脚本：用户回复生成
读取 test.jsonl，对每个样本生成专业回复
"""

import os
import json
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import PtEngine, InferRequest, RequestConfig, get_template
from tqdm import tqdm

# ==================== 配置参数 ====================
# 模型配置
model_id_or_path = 'models/Qwen2.5-1.5B-Instruct/' # Your model path here
checkpoint_path = 'checkpoint/task2_response_generation/final_model'  # 或使用特定的checkpoint
system = 'You are a helpful customer service assistant. Generate appropriate responses to user requests based on their category.'

# 测试数据
test_data_path = './assets/test.jsonl'

# 推理配置
max_new_tokens = 512  # 回复生成需要更长的输出
temperature = 0.7  # 稍微增加一些随机性，使回复更自然
stream = False  # 可以设为True查看生成过程

# 输出配置
output_file = './output/task2_predictions.jsonl'
save_batch_size = 10  # 每处理10个样本保存一次

print("="*80)
print("任务2推理：用户回复生成")
print("="*80)
print(f"模型路径: {model_id_or_path}")
print(f"检查点路径: {checkpoint_path}")
print(f"测试数据: {test_data_path}")
print(f"输出文件: {output_file}")
print(f"Temperature: {temperature}")
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

# 检查数据是否包含category字段
if test_data and 'category' not in test_data[0]:
    print("\n⚠️  警告: test.jsonl 不包含 'category' 字段")
    print("建议先运行 task1_inference.py 获取分类结果，或确保test.jsonl包含category字段")
    response = input("是否继续？(y/n): ")
    if response.lower() != 'y':
        exit(0)

# ==================== 初始化模型 ====================
print(f"\n初始化模型...")
engine = PtEngine(model_id_or_path, adapters=[checkpoint_path])
template = get_template(engine.model_meta.template, engine.processor, default_system=system)
engine.default_template = template
print(f"✓ 模型加载完成")

# ==================== 推理函数 ====================
def generate_response(engine, instruction, category, use_stream=False):
    """
    为单个样本生成回复
    
    Args:
        engine: 推理引擎
        instruction: 用户请求文本
        category: 请求类别
        use_stream: 是否使用流式输出
    
    Returns:
        generated_response: 生成的回复
    """
    # 构造输入（与训练时保持一致）
    query = f"This is a Type {category} user request: {instruction}. Please formulate an appropriate response."
    
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
        print()  # 换行
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
        category = item.get('category', 'UNKNOWN')  # 如果没有category，使用UNKNOWN
        
        if stream:
            print(f"\n样本 {idx+1}/{len(test_data)}:")
            print(f"Category: {category}")
            print(f"Instruction: {instruction[:100]}...")
            print(f"生成回复: ", end='', flush=True)
        
        # 生成回复
        generated_response = generate_response(engine, instruction, category, use_stream=stream)
        
        # 保存结果
        result = {
            'instruction': instruction,
            'category': category,
            'generated_response': generated_response,
            'ground_truth_response': item.get('response', None),  # 如果test.jsonl有标签
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
