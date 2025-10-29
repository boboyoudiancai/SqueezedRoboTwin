import numpy as np
from .dp_model import DP
import yaml

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    """
    Load DP model from checkpoint.

    Supports both standard and squeezed DP checkpoints.

    For standard DP, usr_args should contain:
        - task_name
        - ckpt_setting (dataset config, e.g., 'aloha-agilex_clean')
        - expert_data_num (number of episodes)
        - seed
        - checkpoint_num (epoch number)
        - left_arm_dim, right_arm_dim

    For squeezed DP, additionally include:
        - use_squeezed: True
        - squeeze_strength: e.g., -0.4 (optional, defaults to config file)
        - quantum_limited: True/False (default: False)

    Args:
        usr_args: Dictionary containing checkpoint and model parameters

    Returns:
        DP model instance
    """
    # Calculate action_dim first (needed for config file path)
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2  # 2 gripper

    # Detect if using squeezed DP
    use_squeezed = usr_args.get('use_squeezed', False)
    # Handle string "true"/"false" from command line
    if isinstance(use_squeezed, str):
        use_squeezed = use_squeezed.lower() == 'true'

    # Construct checkpoint path based on policy type
    if use_squeezed:
        # Squeezed DP checkpoint path
        # Use config file default (-0.4) if not provided
        strength = usr_args.get('squeeze_strength', None)
        if strength is None:
            # Load default from config file
            load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
            with open(load_config_path, "r", encoding="utf-8") as f:
                temp_config = yaml.safe_load(f)
            strength = temp_config.get('policy', {}).get('noise_squeezer', {}).get('squeeze_strength', -0.4)
            print(f"[Deploy] Using config default squeeze_strength: {strength}")
        else:
            # Convert string to float if needed
            if isinstance(strength, str):
                strength = float(strength)
            print(f"[Deploy] Using provided squeeze_strength: {strength}")

        quantum_limited = usr_args.get('quantum_limited', False)
        # Handle string "true"/"false"
        if isinstance(quantum_limited, str):
            quantum_limited = quantum_limited.lower() == 'true'

        # Format strength string (same as in eval.sh)
        if strength < 0:
            strength_str = f"n{abs(strength):.2f}".replace('.', '')
        else:
            strength_str = f"p{strength:.2f}".replace('.', '')

        quantum_str = 'q1' if quantum_limited else 'q0'

        ckpt_base = f"checkpoints_squeezed_s{strength_str}_{quantum_str}"
        ckpt_file = (
            f"./policy/DP/{ckpt_base}/"
            f"{usr_args['task_name']}/"
            f"{usr_args['ckpt_setting']}_{usr_args['expert_data_num']}-{usr_args['seed']}/"
            f"{usr_args['checkpoint_num']}.ckpt"
        )
        print(f"[Deploy] Loading Squeezed DP checkpoint:")
        print(f"  Squeeze strength: {strength}")
        print(f"  Quantum limited: {quantum_limited}")
        print(f"  Path: {ckpt_file}")
    else:
        # Standard DP checkpoint path
        ckpt_file = (
            f"./policy/DP/checkpoints/"
            f"{usr_args['task_name']}/"
            f"{usr_args['ckpt_setting']}_{usr_args['expert_data_num']}-{usr_args['seed']}/"
            f"{usr_args['checkpoint_num']}.ckpt"
        )
        print(f"[Deploy] Loading Standard DP checkpoint:")
        print(f"  Path: {ckpt_file}")

    # Load config (both standard and squeezed use same base config)
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']

    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
