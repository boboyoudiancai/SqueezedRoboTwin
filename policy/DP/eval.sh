#!/bin/bash
#
# RoboTwin DP Evaluation Script
# Supports both standard and squeezed DP checkpoints
#
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [use_squeezed] [squeeze_strength] [quantum_limited]
#
# Arguments:
#   task_name         Task name (e.g., beat_block_hammer)
#   task_config       Evaluation config (demo_clean or demo_randomized)
#   ckpt_setting      Checkpoint config (e.g., aloha-agilex_clean)
#   expert_data_num   Number of expert demonstrations used in training
#   seed              Random seed used in training
#   gpu_id            GPU device ID
#   use_squeezed      (Optional) Eval squeezed checkpoint: true/false (default: false)
#   squeeze_strength  (Optional) Squeeze strength used in training (default: -0.8)
#   quantum_limited   (Optional) Quantum limited mode (default: false)
#
# Examples:
#   # Standard DP evaluation
#   bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0
#
#   # Squeezed DP evaluation (auto-detect from ckpt_setting)
#   bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true
#
#   # Squeezed DP with custom strength
#   bash eval.sh beat_block_hammer demo_clean aloha-agilex_clean 50 0 0 true -0.5
#

# Keep unchanged
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
use_squeezed=${7:-false}
squeeze_strength=${8:-}  # Empty = must match training (config default: -0.4)
quantum_limited=${9:-false}

DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33m========================================\033[0m"
echo -e "\033[33m  RoboTwin DP Evaluation\033[0m"
echo -e "\033[33m========================================\033[0m"
echo -e "Task:            ${task_name}"
echo -e "Task config:     ${task_config}"
echo -e "Ckpt setting:    ${ckpt_setting}"
echo -e "Data num:        ${expert_data_num}"
echo -e "Seed:            ${seed}"
echo -e "GPU:             ${gpu_id}"
echo -e "Squeezed mode:   ${use_squeezed}"
if [ "${use_squeezed}" = "true" ]; then
    if [ -z "${squeeze_strength}" ]; then
        echo -e "Squeeze strength: (match training default: -0.4)"
    else
        echo -e "Squeeze strength: ${squeeze_strength}"
    fi
    echo -e "Quantum limited:  ${quantum_limited}"
fi
echo -e "\033[33m========================================\033[0m"

cd ../..

# Build evaluation command
eval_cmd="PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed}"

# Add squeezed parameters if enabled
if [ "${use_squeezed}" = "true" ]; then
    echo -e "\033[32m>>> Evaluating Squeezed DP Checkpoint <<<\033[0m"

    # Determine squeeze_strength for checkpoint path
    # If not provided, use config default -0.4
    eval_squeeze_strength="${squeeze_strength:-"-0.4"}"

    eval_cmd="${eval_cmd} \
    --use_squeezed true \
    --quantum_limited ${quantum_limited}"

    # Only add squeeze_strength if explicitly provided
    if [ -n "${squeeze_strength}" ]; then
        eval_cmd="${eval_cmd} --squeeze_strength ${squeeze_strength}"
    fi

    # Verify checkpoint exists
    if [ "${eval_squeeze_strength}" = "-0.8" ] || [ "${eval_squeeze_strength}" = "-0.80" ]; then
        strength_str="n080"
    elif [ "${eval_squeeze_strength}" = "-0.4" ] || [ "${eval_squeeze_strength}" = "-0.40" ]; then
        strength_str="n040"
    else
        # Format strength string
        strength_abs=$(echo "${eval_squeeze_strength}" | tr -d '-')
        strength_str="n$(echo ${strength_abs} | tr -d '.')"
    fi

    quantum_str="q0"
    [ "${quantum_limited}" = "true" ] && quantum_str="q1"

    ckpt_base="checkpoints_squeezed_s${strength_str}_${quantum_str}"
    ckpt_file="./policy/DP/${ckpt_base}/${task_name}/${ckpt_setting}_${expert_data_num}-${seed}/600.ckpt"

    if [ ! -f "${ckpt_file}" ]; then
        echo -e "\033[31mWARNING: Checkpoint not found: ${ckpt_file}\033[0m"
        echo -e "\033[33mMake sure you trained with squeezed mode and correct parameters\033[0m"
    else
        echo -e "\033[32mCheckpoint found: ${ckpt_file}\033[0m"
    fi
else
    echo -e "\033[32m>>> Evaluating Standard DP Checkpoint <<<\033[0m"

    # Verify standard checkpoint
    ckpt_file="./policy/DP/checkpoints/${task_name}/${ckpt_setting}_${expert_data_num}-${seed}/600.ckpt"
    if [ ! -f "${ckpt_file}" ]; then
        echo -e "\033[31mWARNING: Checkpoint not found: ${ckpt_file}\033[0m"
        echo -e "\033[33mMake sure you trained the model first\033[0m"
    else
        echo -e "\033[32mCheckpoint found: ${ckpt_file}\033[0m"
    fi
fi

# Execute evaluation
echo -e "\033[33mStarting evaluation...\033[0m"
eval ${eval_cmd}