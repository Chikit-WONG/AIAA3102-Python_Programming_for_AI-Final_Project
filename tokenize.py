#!/usr/bin/env python3
"""
使用Qwen2.5分词器对JSONL文件中的instruction和response字段进行分词
"""

import json
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize_jsonl(
    input_file: str,
    output_file: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    instruction_key: str = "instruction",
    response_key: str = "response",
    tokenized_instruction_key: str = "instruction_tokens",
    tokenized_response_key: str = "response_tokens",
):
    """
    读取JSONL文件，对指定字段进行分词，并保存到新文件
    
    参数:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径
        model_name: Qwen2.5模型名称
        instruction_key: instruction字段名
        response_key: response字段名
        tokenized_instruction_key: 分词后的instruction字段名
        tokenized_response_key: 分词后的response字段名
    """
    
    print(f"正在加载分词器: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("分词器加载成功！")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        print("尝试使用本地缓存或其他Qwen2.5模型...")
        # 可以尝试其他模型名称
        alternative_models = [
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ]
        for alt_model in alternative_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(alt_model, trust_remote_code=True)
                print(f"成功加载替代模型: {alt_model}")
                break
            except:
                continue
        else:
            raise RuntimeError("无法加载任何Qwen2.5分词器")
    
    # 统计信息
    total_lines = 0
    processed_lines = 0
    
    print(f"\n正在读取文件: {input_file}")
    
    # 先计算总行数
    with open(input_file, 'r', encoding='gb18030') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"文件共有 {total_lines} 条数据")
    
    # 处理数据
    with open(input_file, 'r', encoding='gb18030') as infile, \
         open(output_file, 'w', encoding='gb18030') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="处理进度"):
            try:
                # 解析JSON
                data = json.loads(line.strip())
                
                # 对instruction进行分词
                if instruction_key in data:
                    instruction_tokens = tokenizer.encode(
                        data[instruction_key],
                        add_special_tokens=False
                    )
                    data[tokenized_instruction_key] = instruction_tokens
                
                # 对response进行分词
                if response_key in data:
                    response_tokens = tokenizer.encode(
                        data[response_key],
                        add_special_tokens=False
                    )
                    data[tokenized_response_key] = response_tokens
                
                # 写入输出文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_lines += 1
                
            except json.JSONDecodeError as e:
                print(f"\n警告: 跳过无效的JSON行: {e}")
                continue
            except Exception as e:
                print(f"\n警告: 处理数据时出错: {e}")
                continue
    
    print(f"\n处理完成!")
    print(f"总行数: {total_lines}")
    print(f"成功处理: {processed_lines}")
    print(f"输出文件: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="使用Qwen2.5分词器对JSONL文件进行分词"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='输入的JSONL文件路径'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='输出的JSONL文件路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/hpc2hdd/home/yuxuanzhao/init_model/Qwen2.5-7B-Instruct/',
        help='Qwen2.5模型名称 (默认: Qwen/Qwen2.5-7B-Instruct)'
    )
    parser.add_argument(
        '--instruction-key',
        type=str,
        default='instruction',
        help='instruction字段名 (默认: instruction)'
    )
    parser.add_argument(
        '--response-key',
        type=str,
        default='response',
        help='response字段名 (默认: response)'
    )
    parser.add_argument(
        '--tokenized-instruction-key',
        type=str,
        default='instruction_tokens',
        help='分词后的instruction字段名 (默认: instruction_tokens)'
    )
    parser.add_argument(
        '--tokenized-response-key',
        type=str,
        default='response_tokens',
        help='分词后的response字段名 (默认: response_tokens)'
    )
    
    args = parser.parse_args()
    
    tokenize_jsonl(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model,
        instruction_key=args.instruction_key,
        response_key=args.response_key,
        tokenized_instruction_key=args.tokenized_instruction_key,
        tokenized_response_key=args.tokenized_response_key,
    )


if __name__ == '__main__':
    main()