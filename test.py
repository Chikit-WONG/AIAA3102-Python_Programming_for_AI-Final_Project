# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# CSV to JSONL Converter
# 将CSV文件转换为JSONL格式，完整保留所有内容
# """

# import csv
# import json
# import sys
# from pathlib import Path


# def csv_to_jsonl(input_csv, output_jsonl=None, encoding='utf-8'):
#     """
#     将CSV文件转换为JSONL格式
    
#     参数:
#         input_csv: 输入的CSV文件路径
#         output_jsonl: 输出的JSONL文件路径（可选，默认为输入文件名.jsonl）
#         encoding: 文件编码（默认utf-8）
#     """
#     # 如果未指定输出文件，使用输入文件名并改后缀为.jsonl
#     if output_jsonl is None:
#         input_path = Path(input_csv)
#         output_jsonl = input_path.with_suffix('.jsonl')
    
#     try:
#         # 读取CSV并转换为JSONL
#         with open(input_csv, 'r', encoding=encoding, newline='') as csv_file:
#             # 使用csv.DictReader自动将每行转换为字典
#             csv_reader = csv.DictReader(csv_file)
            
#             # 写入JSONL文件
#             with open(output_jsonl, 'w', encoding=encoding) as jsonl_file:
#                 row_count = 0
#                 for row in csv_reader:
#                     # 将每行转换为JSON并写入，每行一个JSON对象
#                     json_line = json.dumps(row, ensure_ascii=False)
#                     jsonl_file.write(json_line + '\n')
#                     row_count += 1
        
#         print(f"✓ 转换成功!")
#         print(f"  输入文件: {input_csv}")
#         print(f"  输出文件: {output_jsonl}")
#         print(f"  转换行数: {row_count}")
        
#     except FileNotFoundError:
#         print(f"✗ 错误: 找不到文件 '{input_csv}'")
#         sys.exit(1)
#     except UnicodeDecodeError:
#         print(f"✗ 编码错误: 请尝试使用不同的编码，如 'gbk' 或 'gb18030'")
#         print(f"  使用方法: csv_to_jsonl('{input_csv}', encoding='gbk')")
#         sys.exit(1)
#     except Exception as e:
#         print(f"✗ 转换失败: {e}")
#         sys.exit(1)


# def main():
#     """命令行入口"""
#     if len(sys.argv) < 2:
#         print("使用方法: python csv_to_jsonl.py <input.csv> [output.jsonl] [encoding]")
#         print("示例:")
#         print("  python csv_to_jsonl.py data.csv")
#         print("  python csv_to_jsonl.py data.csv output.jsonl")
#         print("  python csv_to_jsonl.py data.csv output.jsonl gbk")
#         sys.exit(1)
    
#     input_csv = sys.argv[1]
#     output_jsonl = sys.argv[2] if len(sys.argv) > 2 else None
#     encoding = sys.argv[3] if len(sys.argv) > 3 else 'utf-8'
#     print(encoding)
#     csv_to_jsonl(input_csv, output_jsonl, encoding)


# if __name__ == "__main__":
#     main()


# import chardet

# def detect_csv_encoding(file_path):
#     """
#     检测CSV文件的编码类型
    
#     参数:
#         file_path: CSV文件路径
    
#     返回:
#         编码类型字符串
#     """
#     # 读取文件的前几行来检测编码
#     with open(file_path, 'rb') as file:
#         raw_data = file.read(10000)  # 读取前10000字节
    
#     # 使用chardet检测编码
#     result = chardet.detect(raw_data)
#     encoding = result['encoding']
#     confidence = result['confidence']
    
#     print(f"文件: {file_path}")
#     print(f"检测到的编码: {encoding}")
#     print(f"置信度: {confidence:.2%}")
    
#     return encoding

# # 使用示例
# if __name__ == "__main__":
#     file_path = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"  # 替换为你的CSV文件路径
#     encoding = detect_csv_encoding(file_path)

import json

def extract_category_from_jsonl(file_path):
    """
    从JSONL文件中提取所有唯一的category字段值，返回列表（数组）
    
    参数:
        file_path (str): JSONL文件的绝对/相对路径
    
    返回:
        list: 去重后的category值列表，若文件异常则返回空列表
    """
    # 使用集合自动去重，最后转为列表
    category_set = set()

    try:
        # 以UTF-8编码打开文件（兼容中文等特殊字符）
        with open(file_path, 'r', encoding='utf-8') as f:
            # 逐行读取JSONL文件（每行一个JSON对象）
            for line_num, line in enumerate(f, start=1):
                # 去除行首尾空白符，跳过空行
                clean_line = line.strip()
                if not clean_line:
                    continue

                try:
                    # 解析当前行的JSON数据
                    json_data = json.loads(clean_line)
                    
                    # 检查是否包含category字段
                    if 'category' in json_data:
                        # 获取字段值（兼容数字/字符串/布尔等类型）
                        category_val = json_data['category']
                        # 加入集合（自动去重）
                        category_set.add(category_val)
                    else:
                        # 仅警告，不中断程序
                        print(f"【警告】第{line_num}行无category字段，跳过")

                except json.JSONDecodeError as e:
                    # JSON格式错误时的提示
                    print(f"【错误】第{line_num}行JSON格式无效：{str(e)}，跳过")

    except FileNotFoundError:
        print(f"【错误】文件不存在：{file_path}")
    except PermissionError:
        print(f"【错误】无权限访问文件：{file_path}")
    except Exception as e:
        # 捕获其他未知异常
        print(f"【错误】读取文件时发生未知错误：{str(e)}")

    # 将集合转为有序列表（Python3.7+集合保留插入顺序）
    category_list = list(category_set)
    return category_list

# 主程序入口
if __name__ == "__main__":
    # --------------------------
    # 请替换为你的JSONL文件路径
    # --------------------------
    jsonl_file_path = "/hpc2hdd/home/yuxuanzhao/haodong/3102project/assets/test.jsonl"  # 示例："./data/input.jsonl"
    
    # 提取category字段值
    result = extract_category_from_jsonl(jsonl_file_path)
    
    # 打印最终结果（数组格式）
    print("\n=== JSONL文件中category字段的所有值 ===")
    print(result)