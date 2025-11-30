import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse


class ResponseEvaluator:
    """
    响应评估器类，整合三种评估方法
    """
    
    def __init__(self, config):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 设置CUDA设备
        os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda']['visible_devices']
        
        # 初始化ROUGE评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # 加载嵌入模型
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        
        # 加载prompt模板
        with open(config['paths']['prompt_json'], 'r', encoding='utf-8') as f:
            self.prompt_template = json.load(f)
    
    def load_embedding_model(self):
        """加载嵌入模型"""
        if self.embedding_model is None:
            print("正在加载嵌入模型...")
            self.embedding_model = SentenceTransformer(
                self.config['paths']['embedding_model'],
                trust_remote_code=True
            )
            print("嵌入模型加载完成！")
    
    def load_llm_model(self):
        """加载LLM模型"""
        if self.llm_tokenizer is None or self.llm_model is None:
            print(f"正在加载LLM模型: {self.config['paths']['llm_model']}...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config['paths']['llm_model'],
                trust_remote_code=True
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.config['paths']['llm_model'],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("LLM模型加载完成！")
    
    def calculate_rouge_l(self, generated, ground_truth):
        """
        计算ROUGE-L分数
        
        Args:
            generated: 生成的响应
            ground_truth: 标准答案
            
        Returns:
            dict: 包含precision, recall, fmeasure的字典
        """
        try:
            scores = self.rouge_scorer.score(ground_truth, generated)
            return {
                'precision': float(scores['rougeL'].precision),
                'recall': float(scores['rougeL'].recall),
                'fmeasure': float(scores['rougeL'].fmeasure)
            }
        except Exception as e:
            print(f"计算ROUGE-L时出错: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'fmeasure': 0.0
            }
    
    def calculate_cosine_similarity(self, text1, text2):
        """
        计算两个文本的余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 余弦相似度值
        """
        try:
            # 获取文本嵌入
            embedding1 = self.embedding_model.encode([text1])
            embedding2 = self.embedding_model.encode([text2])
            
            # 计算余弦相似度
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"计算余弦相似度时出错: {e}")
            return 0.0
    
    def create_llm_prompt(self, generated_response, ground_truth_response):
        """
        创建LLM评估的prompt，严格遵循prompt.json格式
        
        Args:
            generated_response: 生成的响应
            ground_truth_response: 标准答案
            
        Returns:
            list: 消息列表
        """
        # 构建完整的对话历史
        messages = []
        
        # 添加所有已有的对话（除了最后一个需要填充的user消息）
        for i, msg in enumerate(self.prompt_template[:-1]):
            messages.append(msg)
        
        # 添加最后一个user消息，填充实际的generated_response和ground_truth_response
        final_user_content = f"generated_response: {generated_response}\n\nground_truth_response: {ground_truth_response}"
        messages.append({
            "role": "user",
            "content": final_user_content
        })
        
        return messages
    
    def llm_evaluate(self, generated_response, ground_truth_response):
        """
        使用LLM进行评估
        
        Args:
            generated_response: 生成的响应
            ground_truth_response: 标准答案
            
        Returns:
            str: LLM的评估结果
        """
        try:
            # 创建prompt
            messages = self.create_llm_prompt(generated_response, ground_truth_response)
            
            # 使用tokenizer的chat模板
            prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 生成评估结果
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            
            gen_config = self.config['llm_generation']
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=gen_config['max_new_tokens'],
                    temperature=gen_config['temperature'],
                    top_p=gen_config['top_p'],
                    do_sample=gen_config['do_sample'],
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
        except Exception as e:
            print(f"LLM评估时出错: {e}")
            return f"Error: {str(e)}"
    
    def parse_llm_response(self, llm_response):
        """
        解析LLM的评估结果，提取分数和原因
        
        Args:
            llm_response: LLM的原始响应
            
        Returns:
            dict: 解析后的分数字典
        """
        try:
            result = {
                'total': None,
                'relevance': None,
                'politeness_professionalism': None,
                'guidance_effectiveness': None,
                'reason': None
            }
            
            # 提取total
            if 'total:' in llm_response:
                total_str = llm_response.split('total:')[1].split('.')[0].strip()
                result['total'] = int(total_str)
            
            # 提取relevance
            if 'relevance:' in llm_response:
                relevance_str = llm_response.split('relevance:')[1].split(',')[0].strip()
                result['relevance'] = int(relevance_str)
            
            # 提取politeness and professionalism
            if 'politeness and professionalism:' in llm_response:
                politeness_str = llm_response.split('politeness and professionalism:')[1].split(',')[0].strip()
                result['politeness_professionalism'] = int(politeness_str)
            
            # 提取guidance effectiveness
            if 'guidance effectiveness:' in llm_response:
                guidance_str = llm_response.split('guidance effectiveness:')[1].split('.')[0].strip()
                result['guidance_effectiveness'] = int(guidance_str)
            
            # 提取reason
            if 'Reason:' in llm_response:
                result['reason'] = llm_response.split('Reason:')[1].strip()
            
            return result
        except Exception as e:
            print(f"解析LLM响应时出错: {e}")
            return {
                'total': None,
                'relevance': None,
                'politeness_professionalism': None,
                'guidance_effectiveness': None,
                'reason': llm_response
            }
    
    def evaluate_single(self, generated_response, ground_truth_response):
        """
        对单个响应对进行三种评估
        
        Args:
            generated_response: 生成的响应
            ground_truth_response: 标准答案
            
        Returns:
            dict: 包含三种评估结果的字典
        """
        results = {}
        
        # 1. ROUGE-L评估
        results['rouge_l'] = self.calculate_rouge_l(generated_response, ground_truth_response)
        
        # 2. 余弦相似度评估
        results['cosine_similarity'] = self.calculate_cosine_similarity(
            generated_response,
            ground_truth_response
        )
        
        # 3. LLM评估
        llm_response = self.llm_evaluate(generated_response, ground_truth_response)
        results['llm_evaluation'] = {
            'raw_response': llm_response,
            'parsed_scores': self.parse_llm_response(llm_response)
        }
        
        return results
    
    def process_file(self, input_file, output_file):
        """
        处理整个JSONL文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            
        Returns:
            list: 评估结果列表
        """
        # 加载模型
        self.load_embedding_model()
        self.load_llm_model()
        
        results = []
        
        # 首先读取文件计算总行数
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # 处理文件
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="处理中"):
                # 解析JSON
                data = json.loads(line.strip())
                
                instruction = data.get('instruction', '')
                category = data.get('category', '')
                generated_response = data.get('generated_response', '')
                ground_truth_response = data.get('ground_truth_response', '')
                
                # 评估
                evaluations = self.evaluate_single(generated_response, ground_truth_response)
                
                # 存储结果
                result = {
                    'instruction': instruction,
                    'category': category,
                    'generated_response': generated_response,
                    'ground_truth_response': ground_truth_response,
                    'evaluations': evaluations
                }
                
                results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\n结果已保存到: {output_file}")
        return results


def print_summary(results):
    """
    打印评估结果的统计摘要
    
    Args:
        results: 评估结果列表
    """
    print("\n" + "="*70)
    print("评估结果统计摘要")
    print("="*70)
    
    # 收集统计数据
    rouge_f1_scores = []
    rouge_precision_scores = []
    rouge_recall_scores = []
    cosine_similarities = []
    llm_total_scores = []
    llm_relevance_scores = []
    llm_politeness_scores = []
    llm_guidance_scores = []
    
    for result in results:
        evals = result['evaluations']
        
        # ROUGE-L
        rouge_f1_scores.append(evals['rouge_l']['fmeasure'])
        rouge_precision_scores.append(evals['rouge_l']['precision'])
        rouge_recall_scores.append(evals['rouge_l']['recall'])
        
        # 余弦相似度
        cosine_similarities.append(evals['cosine_similarity'])
        
        # LLM评估
        llm_parsed = evals['llm_evaluation']['parsed_scores']
        if llm_parsed['total'] is not None:
            llm_total_scores.append(llm_parsed['total'])
        if llm_parsed['relevance'] is not None:
            llm_relevance_scores.append(llm_parsed['relevance'])
        if llm_parsed['politeness_professionalism'] is not None:
            llm_politeness_scores.append(llm_parsed['politeness_professionalism'])
        if llm_parsed['guidance_effectiveness'] is not None:
            llm_guidance_scores.append(llm_parsed['guidance_effectiveness'])
    
    print(f"\n总记录数: {len(results)}")
    
    # ROUGE-L统计
    print("\n" + "-"*70)
    print("1. ROUGE-L 评估统计:")
    print(f"  F1 分数:")
    print(f"    平均值: {np.mean(rouge_f1_scores):.4f}")
    print(f"    中位数: {np.median(rouge_f1_scores):.4f}")
    print(f"    最小值: {min(rouge_f1_scores):.4f}")
    print(f"    最大值: {max(rouge_f1_scores):.4f}")
    print(f"    标准差: {np.std(rouge_f1_scores):.4f}")
    print(f"  Precision 平均值: {np.mean(rouge_precision_scores):.4f}")
    print(f"  Recall 平均值: {np.mean(rouge_recall_scores):.4f}")
    
    # 余弦相似度统计
    print("\n" + "-"*70)
    print("2. 余弦相似度评估统计:")
    print(f"  平均值: {np.mean(cosine_similarities):.4f}")
    print(f"  中位数: {np.median(cosine_similarities):.4f}")
    print(f"  最小值: {min(cosine_similarities):.4f}")
    print(f"  最大值: {max(cosine_similarities):.4f}")
    print(f"  标准差: {np.std(cosine_similarities):.4f}")
    
    # LLM评估统计
    print("\n" + "-"*70)
    print("3. LLM评估统计:")
    if llm_total_scores:
        print(f"  总分 (满分10):")
        print(f"    平均值: {np.mean(llm_total_scores):.2f}")
        print(f"    中位数: {np.median(llm_total_scores):.2f}")
        print(f"    最小值: {min(llm_total_scores)}")
        print(f"    最大值: {max(llm_total_scores)}")
        print(f"    标准差: {np.std(llm_total_scores):.2f}")
    
    if llm_relevance_scores:
        print(f"\n  相关性 (满分5):")
        print(f"    平均值: {np.mean(llm_relevance_scores):.2f}")
        print(f"    标准差: {np.std(llm_relevance_scores):.2f}")
    
    if llm_politeness_scores:
        print(f"\n  礼貌专业性 (满分3):")
        print(f"    平均值: {np.mean(llm_politeness_scores):.2f}")
        print(f"    标准差: {np.std(llm_politeness_scores):.2f}")
    
    if llm_guidance_scores:
        print(f"\n  指导有效性 (满分2):")
        print(f"    平均值: {np.mean(llm_guidance_scores):.2f}")
        print(f"    标准差: {np.std(llm_guidance_scores):.2f}")
    
    print("\n" + "="*70)


def load_config(config_file):
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Response Evaluation Script')
    parser.add_argument('--config', type=str, default='./eval/config.json',
                        help='配置文件路径')
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件路径（覆盖配置文件）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（覆盖配置文件）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.input:
        config['paths']['input_file'] = args.input
    if args.output:
        config['paths']['output_file'] = args.output
    
    # 创建评估器
    evaluator = ResponseEvaluator(config)
    
    # 处理文件
    results = evaluator.process_file(
        config['paths']['input_file'],
        config['paths']['output_file']
    )
    
    # 打印统计摘要
    print_summary(results)


if __name__ == "__main__":
    main()