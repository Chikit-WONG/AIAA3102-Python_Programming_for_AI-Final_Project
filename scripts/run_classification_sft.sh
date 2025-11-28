#!/bin/bash
#SBATCH -p debug        # 指定GPU队列
#SBATCH -o ./temp/output.txt  # 指定作业标准输出文件，%j为作业号  SBATCH -o output_%j.txt
#SBATCH -e ./temp/err.txt    # 指定作业标准错误输出文件  SBATCH -e err_%j.txt
#SBATCH -n 2            # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH -D .        # 指定作业执行路径为当前目录

# 加载CUDA模块（如果需要）
module load cuda/12.4

# 激活 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate showo1

# # 设置 LD_LIBRARY_PATH，只使用 conda 下的 .so 文件
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# # 可选调试：查看 .so 是否在位
# echo "[DEBUG] Verifying libcudnn_graph.so presence:"
# ls -l $CONDA_PREFIX/lib/libcudnn_graph.so*

results_output_path=./results
folder_name=run2
results_output_path_folder="$results_output_path/$folder_name"
# 创建结果输出目录（如果不存在）
mkdir -p "$results_output_path"
mkdir -p "$results_output_path_folder"
mkdir -p "$results_output_path_folder/main_result_and_logs"
mkdir -p "$results_output_path_folder/spatial_graphs"
mkdir -p "$results_output_path_folder/images"

# ~/dataset/EgoTextQA/data/egotextvqa_indoor/total_frame.json \
# --output_path ./results/outputs/qwen2_spatial \
# --graph_output_dir ./results/spatial_graphs \
# --image_output_path /hpc2hdd/home/ypan477/workdir/spatialGraph/ckwong/results/images \
# --low_bound 0
# --upper_bound 1
# Job 执行主体
echo "Job started at $(date)"
python spatial_infer_qwen2.py \
    --output_path $results_output_path_folder/main_result_and_logs \
    --graph_output_dir $results_output_path_folder/spatial_graphs \
    --image_output_path $results_output_path_folder/images \
    --json_path ./test-use_enhance.json \
    --image_root ~/dataset/EgoTextQA/indoor/fps6_frames \
    --model_path ../models/Qwen2-VL-7B-Instruct \
    --batch_size 4 \
    --model_name Qwen2-spatial \
    --task_type indoor \
    --max_objects 8 \
    --graph_base_model_path ../models/Qwen2.5-VL-7B-Instruct \
    --depth_model_path ../models/Depth-Anything-V2-Base/depth_anything_v2_vitb.pth \
    --temp_dir ./temp \
    --prompts_path ./prompts/prompts.yaml \
    --save_json \
    --visualize 
echo "Job ended at $(date)"

# 退出环境
conda deactivate