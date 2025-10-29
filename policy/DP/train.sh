#!/bin/bash
#
# RoboTwin Diffusion Policy Training Script
# Supports both standard DDPM and squeezed diffusion modes
#
# Usage:
#   bash train.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [use_squeezed] [squeeze_strength]
#
# Arguments:
#   task_name         Task name (e.g., beat_block_hammer)
#   task_config       Config: aloha-agilex_clean, franka-panda_randomized, etc.
#   expert_data_num   Number of expert demonstrations (e.g., 50)
#   seed              Random seed
#   action_dim        Action space dimension (14 or 16)
#   gpu_id            GPU device ID
#   use_squeezed      (Optional) Enable squeezed diffusion: true/false (default: false)
#   squeeze_strength  (Optional) Squeeze strength (default: -0.8, range: [-1.5, 0.0])
#
# Examples:
#   # Standard DDPM
#   bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0
#
#   # Squeezed DDPM (auto PCA dir)
#   bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true
#
#   # Squeezed DDPM with custom strength
#   bash train.sh beat_block_hammer aloha-agilex_clean 50 0 14 0 true -0.5
#

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
use_squeezed=${7:-false}
squeeze_strength=${8:-}  # Empty = use config file default (-0.4)

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33m========================================\033[0m"
echo -e "\033[33m  RoboTwin DP Training\033[0m"
echo -e "\033[33m========================================\033[0m"
echo -e "Task:            ${task_name}"
echo -e "Config:          ${task_config}"
echo -e "Data num:        ${expert_data_num}"
echo -e "Seed:            ${seed}"
echo -e "Action dim:      ${action_dim}"
echo -e "GPU:             ${gpu_id}"
echo -e "Squeezed mode:   ${use_squeezed}"
if [ "${use_squeezed}" = "true" ]; then
    if [ -z "${squeeze_strength}" ]; then
        echo -e "Squeeze strength: (config default: -0.4)"
    else
        echo -e "Squeeze strength: ${squeeze_strength}"
    fi
fi
echo -e "\033[33m========================================\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 检查zarr文件是否存在
if [ ! -d "/home/wangbo/data/dataset/${task_name}/${task_config}_${expert_data_num}.zarr" ]; then
    echo -e "\033[33mZarr not found, processing data...\033[0m"
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

# Build base training command
train_cmd="python train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path=/home/wangbo/data/dataset/${task_name}/${task_config}_${expert_data_num}.zarr \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device=cuda:0 \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=$head_camera_type"

# Add squeezed diffusion parameters if enabled
if [ "${use_squeezed}" = "true" ]; then
    echo -e "\033[32m>>> Squeezed Diffusion Enabled <<<\033[0m"

    # Auto-construct PCA directory path
    pca_base="/home/wangbo/RoboTwin/pca/results"
    pca_dir="${pca_base}/${task_name}-${task_config}_${expert_data_num}"

    # Validate PCA directory
    if [ ! -d "${pca_dir}" ]; then
        echo -e "\033[31mERROR: PCA dir not found: ${pca_dir}\033[0m"
        echo -e "\033[33mRun PCA analysis first:\033[0m"
        echo "  cd /home/wangbo/RoboTwin"
        echo "  python pca/pca_analysis.py /home/wangbo/data/dataset/${task_name}/${task_config}_${expert_data_num}.zarr --output_dir ${pca_base}"
        exit 1
    fi

    if [ ! -f "${pca_dir}/pca_components_frames.npy" ] || [ ! -f "${pca_dir}/pca_variance_frames.npy" ]; then
        echo -e "\033[31mERROR: PCA files missing in ${pca_dir}\033[0m"
        exit 1
    fi

    echo -e "\033[32mPCA dir: ${pca_dir}\033[0m"

    # Build squeezed parameters
    train_cmd="${train_cmd} \
        policy.noise_squeezer._target_=diffusion_policy.model.diffusion.noise_squeezer.NoiseSqueezer \
        +policy.noise_squeezer.pca_dir=${pca_dir} \
        policy.noise_squeezer.quantum_limited=false"

    # Only override squeeze_strength if explicitly provided
    if [ -n "${squeeze_strength}" ]; then
        train_cmd="${train_cmd} policy.noise_squeezer.squeeze_strength=${squeeze_strength}"
    fi
else
    echo -e "\033[32m>>> Standard DDPM Mode <<<\033[0m"
fi

# Execute training
echo -e "\033[33mStarting training...\033[0m"
eval ${train_cmd}