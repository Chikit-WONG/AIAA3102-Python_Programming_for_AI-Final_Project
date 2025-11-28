import json
import random
from collections import defaultdict, Counter
import argparse

def load_data(file_path):
    """加载JSONL文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def stratified_split(data, test_ratio=0.1, random_seed=42):
    """分层抽样划分数据集"""
    random.seed(random_seed)
    
    # 按category分组
    category_groups = defaultdict(list)
    for item in data:
        category = item.get('category', 'UNKNOWN')
        category_groups[category].append(item)
    
    train_data = []
    val_data = []
    
    # 对每个category进行抽样
    for category, items in category_groups.items():
        random.shuffle(items)
        split_index = int(len(items) * (1 - test_ratio))
        
        # 确保验证集至少有一个样本（如果数据量很少）
        if split_index == len(items):
            split_index = len(items) - 1
        
        train_data.extend(items[:split_index])
        val_data.extend(items[split_index:])
    
    return train_data, val_data, category_groups

def calculate_statistics(data, category_groups):
    """计算统计信息"""
    # 计算每个category的数量
    category_counts = Counter()
    category_intent_map = defaultdict(set)
    
    for item in data:
        category = item.get('category', 'UNKNOWN')
        intent = item.get('intent', 'UNKNOWN')
        category_counts[category] += 1
        category_intent_map[category].add(intent)
    
    # 计算每个category的intent数量
    category_intent_counts = {}
    for category, intents in category_intent_map.items():
        category_intent_counts[category] = len(intents)
    
    return category_counts, category_intent_counts, category_intent_map

def save_data(data, file_path):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='数据集分层抽样划分')
    parser.add_argument('--input_file', type=str, required=True, help='输入JSONL文件路径')
    parser.add_argument('--train_file', type=str, default='val.jsonl', help='训练集输出文件路径')
    parser.add_argument('--val_file', type=str, default='test.jsonl', help='验证集输出文件路径')
    parser.add_argument('--test_ratio', type=float, default=0.67, help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 加载数据
    print("正在加载数据...")
    data = load_data(args.input_file)
    print(f"总共加载 {len(data)} 条数据")
    
    # 分层抽样划分
    print("正在进行分层抽样划分...")
    train_data, val_data, category_groups = stratified_split(data, args.test_ratio, args.random_seed)
    
    # 保存划分后的数据
    print("正在保存划分后的数据...")
    save_data(train_data, args.train_file)
    save_data(val_data, args.val_file)
    
    # 计算统计信息
    print("\n=== 数据集统计信息 ===")
    
    # 总体统计
    total_category_counts, total_intent_counts, total_intent_map = calculate_statistics(data, category_groups)
    train_category_counts, train_intent_counts, _ = calculate_statistics(train_data, category_groups)
    val_category_counts, val_intent_counts, _ = calculate_statistics(val_data, category_groups)
    
    print(f"\n总体统计:")
    print(f"总数据量: {len(data)}")
    print(f"训练集数量: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"验证集数量: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    
    print(f"\nCategory分布 (总体):")
    print("-" * 50)
    print(f"{'Category':<20} {'数量':<8} {'比例':<8} {'Intent数量':<12}")
    print("-" * 50)
    for category in sorted(total_category_counts.keys()):
        count = total_category_counts[category]
        ratio = count / len(data) * 100
        intent_count = total_intent_counts[category]
        print(f"{category:<20} {count:<8} {ratio:<7.1f}% {intent_count:<12}")
    
    print(f"\nCategory分布 (训练集):")
    print("-" * 50)
    print(f"{'Category':<20} {'数量':<8} {'比例':<8} {'Intent数量':<12}")
    print("-" * 50)
    for category in sorted(train_category_counts.keys()):
        count = train_category_counts[category]
        ratio = count / len(train_data) * 100
        intent_count = train_intent_counts[category]
        print(f"{category:<20} {count:<8} {ratio:<7.1f}% {intent_count:<12}")
    
    print(f"\nCategory分布 (验证集):")
    print("-" * 50)
    print(f"{'Category':<20} {'数量':<8} {'比例':<8} {'Intent数量':<12}")
    print("-" * 50)
    for category in sorted(val_category_counts.keys()):
        count = val_category_counts[category]
        ratio = count / len(val_data) * 100
        intent_count = val_intent_counts[category]
        print(f"{category:<20} {count:<8} {ratio:<7.1f}% {intent_count:<12}")
    
    # 输出每个category包含的intent详情
    print(f"\n=== 各Category包含的Intent详情 ===")
    for category in sorted(total_intent_map.keys()):
        intents = sorted(list(total_intent_map[category]))
        print(f"\n{category} ({len(intents)}个intent):")
        print(f"  {', '.join(intents)}")
    
    print(f"\n划分完成!")
    print(f"训练集已保存至: {args.train_file}")
    print(f"验证集已保存至: {args.val_file}")

if __name__ == "__main__":
    main()