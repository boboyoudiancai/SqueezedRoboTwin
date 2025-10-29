import numpy as np
try:
    from .dp_model import DP
except:
    pass

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
    Load DP model from checkpoint for double environment evaluation.

    Supports both standard and squeezed DP checkpoints.

    Args:
        usr_args: Dictionary containing:
            - task_name
            - ckpt_setting
            - expert_data_num
            - seed
            - checkpoint_num
            - use_squeezed (optional): True for squeezed DP
            - squeeze_strength (optional): e.g., -0.8
            - quantum_limited (optional): True/False

    Returns:
        DP model instance
    """
    # Detect if using squeezed DP
    use_squeezed = usr_args.get('use_squeezed', False)

    # Construct checkpoint path based on policy type
    if use_squeezed:
        # Squeezed DP checkpoint path
        strength = usr_args.get('squeeze_strength', -0.8)
        quantum_limited = usr_args.get('quantum_limited', False)

        # Format strength string
        if strength < 0:
            strength_str = f"n{abs(strength):.2f}".replace('.', '')
        else:
            strength_str = f"p{strength:.2f}".replace('.', '')

        quantum_str = 'q1' if quantum_limited else 'q0'

        ckpt_base = f"checkpoints_squeezed_s{strength_str}_{quantum_str}"
        ckpt_file = (
            f"./policy/DP/{ckpt_base}/"
            f"{usr_args['task_name']}-{usr_args['ckpt_setting']}-"
            f"{usr_args['expert_data_num']}-{usr_args['seed']}/"
            f"{usr_args['checkpoint_num']}.ckpt"
        )
        print(f"[Deploy Double Env] Loading Squeezed DP: strength={strength}, quantum={quantum_limited}")
    else:
        # Standard DP checkpoint path
        ckpt_file = (
            f"./policy/DP/checkpoints/"
            f"{usr_args['task_name']}-{usr_args['ckpt_setting']}-"
            f"{usr_args['expert_data_num']}-{usr_args['seed']}/"
            f"{usr_args['checkpoint_num']}.ckpt"
        )
        print(f"[Deploy Double Env] Loading Standard DP")

    print(f"  Checkpoint: {ckpt_file}")
    return DP(ckpt_file)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.call(func_name='get_action', obs=obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.call(func_name='update_obs', obs=obs)
