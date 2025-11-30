"""
邮件智能助手后端API (整合版)
整合task1（分类）和task2（回复生成）两个微调模型
同时托管前端页面，方便共享访问
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

# 设置GPU（根据需要调整）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import PtEngine, InferRequest, RequestConfig, get_template

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)  # 允许跨域请求

# ==================== 配置参数 ====================
# 服务器配置
HOST = '0.0.0.0'  # 允许外部访问
PORT = 5000       # 端口号

# 基础模型路径
BASE_MODEL_PATH = '/hpc2hdd/home/yuxuanzhao/init_model/Qwen2.5-1.5B-Instruct/'

# Task1: 分类模型
TASK1_CHECKPOINT = 'checkpoint/task1_classification/final_model'
TASK1_SYSTEM = 'You are a helpful assistant specialized in classifying user requests.'

# Task2: 回复生成模型
TASK2_CHECKPOINT = 'checkpoint/task2_response_generation/final_model'
TASK2_SYSTEM = 'You are a helpful customer service assistant. Generate appropriate responses to user requests based on their category.'

# 推理配置
TASK1_MAX_TOKENS = 128
TASK1_TEMPERATURE = 0

TASK2_MAX_TOKENS = 512
TASK2_TEMPERATURE = 0.7

# ==================== 全局模型引擎 ====================
task1_engine = None
task2_engine = None

def init_models():
    """初始化两个模型引擎"""
    global task1_engine, task2_engine
    
    print("="*60)
    print("正在初始化模型...")
    print("="*60)
    
    # 初始化Task1分类模型
    print("\n[1/2] 加载分类模型 (Task1)...")
    task1_engine = PtEngine(BASE_MODEL_PATH, adapters=[TASK1_CHECKPOINT])
    task1_template = get_template(
        task1_engine.model_meta.template, 
        task1_engine.processor, 
        default_system=TASK1_SYSTEM
    )
    task1_engine.default_template = task1_template
    print("✓ 分类模型加载完成")
    
    # 初始化Task2回复生成模型
    print("\n[2/2] 加载回复生成模型 (Task2)...")
    task2_engine = PtEngine(BASE_MODEL_PATH, adapters=[TASK2_CHECKPOINT])
    task2_template = get_template(
        task2_engine.model_meta.template, 
        task2_engine.processor, 
        default_system=TASK2_SYSTEM
    )
    task2_engine.default_template = task2_template
    print("✓ 回复生成模型加载完成")
    
    print("\n" + "="*60)
    print("所有模型初始化完成！服务已就绪")
    print("="*60 + "\n")

def classify_email(instruction):
    """使用Task1模型对邮件进行分类"""
    query = f"Please classify the following user request into the appropriate category: {instruction}"
    
    infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
    request_config = RequestConfig(
        max_tokens=TASK1_MAX_TOKENS, 
        temperature=TASK1_TEMPERATURE, 
        stream=False
    )
    
    resp_list = task1_engine.infer([infer_request], request_config)
    category = resp_list[0].choices[0].message.content.strip()
    
    return category

def generate_response(instruction, category):
    """使用Task2模型根据分类生成回复"""
    query = f"This is a Type {category} user request: {instruction}. Please formulate an appropriate response."
    
    infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
    request_config = RequestConfig(
        max_tokens=TASK2_MAX_TOKENS, 
        temperature=TASK2_TEMPERATURE, 
        stream=False
    )
    
    resp_list = task2_engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content.strip()
    
    return response

# ==================== 前端页面路由 ====================

@app.route('/')
def index():
    """提供前端页面"""
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory(BASE_DIR, filename)

# ==================== API 路由 ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'task1_loaded': task1_engine is not None,
        'task2_loaded': task2_engine is not None
    })

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """分类接口"""
    try:
        data = request.get_json()
        email_content = data.get('email', '').strip()
        
        if not email_content:
            return jsonify({'error': '邮件内容不能为空'}), 400
        
        category = classify_email(email_content)
        
        return jsonify({
            'success': True,
            'category': category
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """回复生成接口"""
    try:
        data = request.get_json()
        email_content = data.get('email', '').strip()
        category = data.get('category', '').strip()
        
        if not email_content:
            return jsonify({'error': '邮件内容不能为空'}), 400
        if not category:
            return jsonify({'error': '类别不能为空'}), 400
        
        response = generate_response(email_content, category)
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def api_process():
    """完整处理接口 - 先分类再生成回复"""
    try:
        data = request.get_json()
        email_content = data.get('email', '').strip()
        
        if not email_content:
            return jsonify({'error': '邮件内容不能为空'}), 400
        
        # Step 1: 分类
        category = classify_email(email_content)
        
        # Step 2: 生成回复
        response = generate_response(email_content, category)
        
        return jsonify({
            'success': True,
            'category': category,
            'response': response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== 获取服务器IP ====================
def get_local_ip():
    """获取本机IP地址"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 初始化模型
    init_models()
    
    # 获取本机IP
    local_ip = get_local_ip()
    
    # 启动服务
    print("\n" + "="*60)
    print("🚀 智能邮件助手服务已启动！")
    print("="*60)
    print(f"\n📍 本地访问地址:")
    print(f"   http://localhost:{PORT}")
    print(f"\n📍 局域网访问地址 (其他电脑使用这个):")
    print(f"   http://{local_ip}:{PORT}")
    print(f"\n💡 提示: 确保防火墙允许 {PORT} 端口")
    print("="*60)
    print("\n可用接口:")
    print(f"  - GET  http://{local_ip}:{PORT}/           - 前端页面")
    print(f"  - GET  http://{local_ip}:{PORT}/api/health - 健康检查")
    print(f"  - POST http://{local_ip}:{PORT}/api/classify - 邮件分类")
    print(f"  - POST http://{local_ip}:{PORT}/api/generate - 回复生成")
    print(f"  - POST http://{local_ip}:{PORT}/api/process  - 完整处理")
    print("\n" + "="*60 + "\n")
    
    app.run(host=HOST, port=PORT, debug=False, threaded=True)