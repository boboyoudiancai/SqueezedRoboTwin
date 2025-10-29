import sys
sys.path.append('/home/wangbo/RoboTwin')
sys.path.append('/home/wangbo/RoboTwin/policy/DP')

import os
import argparse
import zarr
import numpy as np
from sklearn.decomposition import PCA
from diffusion_policy.model.common.normalizer import LinearNormalizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import yaml
import subprocess
import importlib
from pathlib import Path
from envs import CONFIGS_PATH

# ========== 日志输出类 ==========
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        self.closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self.closed:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.closed:
            self.log.flush()

    def close(self):
        if not self.closed:
            self.log.close()
            self.closed = True

# ========== 参数解析 ==========
parser = argparse.ArgumentParser(description='RoboTwin Action PCA Analysis')
parser.add_argument('zarr_path', type=str, help='Path to zarr dataset (e.g., /path/to/data.zarr)')
parser.add_argument('--output_dir', type=str, default='/home/wangbo/RoboTwin/pca/results',
                    help='Output directory for results (default: /home/wangbo/RoboTwin/pca/results)')
parser.add_argument('--visualize', action='store_true',
                    help='Generate simulation videos after PCA analysis')
parser.add_argument('--task_config', type=str, default='demo_clean',
                    help='Task config for visualization (demo_clean or demo_randomized)')
parser.add_argument('--vis_seed', type=int, default=0,
                    help='Random seed for visualization')
parser.add_argument('--framewise_steps', type=int, default=100,
                    help='Number of steps per direction for framewise PCA visualization')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device to use (default: 0, can specify multiple like "0,1")')
args = parser.parse_args()

# ========== 设置GPU ==========
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(f"使用GPU: {args.gpu}")

zarr_path = args.zarr_path
output_base_dir = args.output_dir

# 提取数据集名称和任务名
# 完整路径格式: /home/wangbo/data/dataset/{task_name}/{embodiment}_{config}_{episodes}.zarr
# 例如: /home/wangbo/data/dataset/beat_block_hammer/arx-x5_randomized_500.zarr
zarr_path_obj = Path(zarr_path)

# 检查zarr路径是否存在
if not zarr_path_obj.exists():
    print(f"❌ 错误: Zarr文件不存在: {zarr_path}")
    sys.exit(1)

zarr_filename = zarr_path_obj.stem  
task_name_from_path = zarr_path_obj.parent.name  

# 检查是否能提取到任务名（防止zarr文件直接在根目录）
if task_name_from_path == '.' or task_name_from_path == '':
    print(f"⚠️  警告: 无法从路径中提取任务名，使用zarr文件名作为数据集标识")
    print(f"    建议使用标准路径格式: /path/to/{{task_name}}/{{embodiment}}_{{config}}_{{episodes}}.zarr")
    dataset_name = zarr_filename
    task_name_extracted = None
else:
    # 组合成完整的数据集标识（包含任务名）
    # 格式: {task_name}-{embodiment}_{config}_{episodes}
    dataset_name = f"{task_name_from_path}-{zarr_filename}"
    task_name_extracted = task_name_from_path

# 创建输出文件夹
output_dir = os.path.join(output_base_dir, dataset_name)
os.makedirs(output_dir, exist_ok=True)

# 设置日志输出
log_file = os.path.join(output_dir, 'pca_analysis_log.txt')
logger = Logger(log_file)
sys.stdout = logger

print("=" * 60)
print("RoboTwin Action PCA Analysis")
print("=" * 60)
print(f"数据集: {dataset_name}")
print(f"输出目录: {output_dir}")
print(f"使用GPU: {args.gpu}")
print("=" * 60)

# ========== 数据加载 ==========
print("\n[1/5] 加载数据...")
print(f"  - Zarr路径: {zarr_path}")
root = zarr.open(zarr_path, 'r')
actions = root['data/action'][:]  
episode_ends = root['meta/episode_ends'][:]

print(f"  - 总帧数: {actions.shape[0]}")
print(f"  - Action维度: {actions.shape[1]}")
print(f"  - Episode数: {len(episode_ends)}")

# ========== DP归一化 (gaussian模式) ==========
print("\n[2/5] DP Gaussian归一化 (zero-mean, unit-variance)...")
normalizer = LinearNormalizer()
normalizer.fit(data=actions, last_n_dims=1, mode='gaussian')
actions_norm = normalizer.normalize(actions)  

print(f"  - 归一化后均值: {actions_norm.mean(axis=0).mean():.6f}")
print(f"  - 归一化后标准差: {actions_norm.std(axis=0).mean():.6f}")

# ========== 方案1: 时间序列PCA ==========
print("\n[3/5] 方案1: 时间序列PCA...")
episode_starts = np.concatenate([[0], episode_ends[:-1]])
T_max = int((episode_ends - episode_starts).max())
action_dim = actions.shape[1]
print(f"  - T_max (最大episode长度): {T_max}")

episodes = []
for i in range(len(episode_ends)):
    start, end = int(episode_starts[i]), int(episode_ends[i])
    ep_data = actions_norm[start:end]
    padded = np.zeros((T_max, action_dim))
    padded[:len(ep_data)] = ep_data
    episodes.append(padded)

X_temporal = np.array(episodes).reshape(len(episode_ends), -1)  
print(f"  - 输入形状: {X_temporal.shape}")

pca_temporal = PCA()
pca_temporal.fit(X_temporal)
components_temporal = pca_temporal.components_
explained_variance_temporal = pca_temporal.explained_variance_ratio_

print(f"  - PCA components形状: {components_temporal.shape}")
print(f"  - 前5个主成分方差解释率: {explained_variance_temporal[:5]}")
print(f"  - 累积方差解释率 (前10个): {explained_variance_temporal[:10].sum():.4f}")

# 生成热力图：第一主成分载荷矩阵 (T_max × action_dim)
print(f"\n  生成热力图: 第一主成分载荷矩阵 (时间×动作)")
pc1_temporal = components_temporal[0].reshape(T_max, action_dim)  
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(pc1_temporal, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Action Dimension', fontsize=12)
ax.set_ylabel('Time Step', fontsize=12)
ax.set_title(f'Temporal PCA: PC1 Loadings (Var Explained: {explained_variance_temporal[0]:.2%})', fontsize=14)
plt.colorbar(im, ax=ax, label='Loading Weight')
heatmap_path_temporal = os.path.join(output_dir, 'pca_heatmap_temporal.png')
plt.savefig(heatmap_path_temporal, dpi=150, bbox_inches='tight')
plt.close()
print(f"   热力图已保存: pca_heatmap_temporal.png")

# 图3: 方案1散点图 - Episode相似性
print(f"\n  生成散点图: Episode在PC空间的分布")
num_episodes = len(episode_ends)
Z_temporal = X_temporal @ components_temporal.T  # (num_episodes, num_episodes)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(Z_temporal[:, 0], Z_temporal[:, 1], s=100, alpha=0.6, c=range(num_episodes), cmap='viridis')
# 只标注部分episode ID，避免过于拥挤
annotation_step = max(1, num_episodes // 20)  # 最多显示20个标签
for i in range(0, num_episodes, annotation_step):
    ax.annotate(str(i), (Z_temporal[i, 0], Z_temporal[i, 1]), fontsize=8, ha='center')
ax.set_xlabel(f'PC1 ({explained_variance_temporal[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance_temporal[1]:.1%})', fontsize=12)
ax.set_title(f'Temporal PCA: Episode Similarity ({num_episodes} trajectories)', fontsize=14)
plt.colorbar(scatter, ax=ax, label='Episode ID')
scatter_path_temporal = os.path.join(output_dir, 'pca_scatter_temporal.png')
plt.savefig(scatter_path_temporal, dpi=150, bbox_inches='tight')
plt.close()
print(f"   散点图已保存: pca_scatter_temporal.png")

# ========== 方案2: 所有帧PCA ==========
print("\n[4/5] 方案2: 所有帧PCA...")
print(f"  - 输入形状: {actions_norm.shape}")

pca_frames = PCA()
pca_frames.fit(actions_norm)
components_frames = pca_frames.components_
explained_variance_frames = pca_frames.explained_variance_ratio_
explained_variance_frames_raw = pca_frames.explained_variance_  

print(f"  - PCA components形状: {components_frames.shape}")
print(f"  - 所有主成分方差解释率: {explained_variance_frames}")
print(f"  - 累积方差解释率: {explained_variance_frames.sum():.4f}")
print(f"  - PC1原始方差: {explained_variance_frames_raw[0]:.6f}")

# 生成热力图：前N个主成分载荷矩阵
print(f"\n  生成热力图: 前10个主成分载荷矩阵")
n_components_show = min(10, action_dim)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(components_frames[:n_components_show, :], aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Action Dimension', fontsize=12)
ax.set_ylabel('Principal Component', fontsize=12)
ax.set_title('Frame-wise PCA: Top 10 PC Loadings', fontsize=14)
ax.set_yticks(range(n_components_show))
ax.set_yticklabels([f'PC{i+1}\n({explained_variance_frames[i]:.1%})' for i in range(n_components_show)])
plt.colorbar(im, ax=ax, label='Loading Weight')
heatmap_path_frames = os.path.join(output_dir, 'pca_heatmap_frames.png')
plt.savefig(heatmap_path_frames, dpi=150, bbox_inches='tight')
plt.close()
print(f"   热力图已保存: pca_heatmap_frames.png")

# 图1: 方案2散点图 - 动作空间分布
print(f"\n  生成散点图: 动作在PC空间的分布")
Z_frames = actions_norm @ components_frames.T  
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(Z_frames[:, 0], Z_frames[:, 1],
                    c=np.arange(len(Z_frames)),
                    cmap='viridis', s=5, alpha=0.5)
ax.set_xlabel(f'PC1 ({explained_variance_frames[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance_frames[1]:.1%})', fontsize=12)
ax.set_title('Frame-wise PCA: Action Space Distribution', fontsize=14)
plt.colorbar(scatter, ax=ax, label='Frame Index')
scatter_path_frames = os.path.join(output_dir, 'pca_scatter_frames.png')
plt.savefig(scatter_path_frames, dpi=150, bbox_inches='tight')
plt.close()
print(f"   散点图已保存: pca_scatter_frames.png")

# 图2: 方案2时间曲线 - 每个episode的PC1演化
print(f"\n  生成时间曲线: PC1随时间演化 (50条轨迹)")
fig, ax = plt.subplots(figsize=(14, 6))
for ep in range(len(episode_ends)):
    start = 0 if ep == 0 else int(episode_ends[ep-1])
    end = int(episode_ends[ep])
    Z_ep = Z_frames[start:end, 0]  # PC1
    ax.plot(Z_ep, alpha=0.3, color='blue', linewidth=1)
ax.set_xlabel('Time Step (within episode)', fontsize=12)
ax.set_ylabel(f'PC1 Value ({explained_variance_frames[0]:.1%})', fontsize=12)
ax.set_title('Frame-wise PCA: PC1 Evolution (50 episodes overlaid)', fontsize=14)
ax.grid(True, alpha=0.3)
timecurve_path_frames = os.path.join(output_dir, 'pca_timecurve_frames.png')
plt.savefig(timecurve_path_frames, dpi=150, bbox_inches='tight')
plt.close()
print(f"   时间曲线已保存: pca_timecurve_frames.png")

# ========== 保存结果 ==========
print("\n[5/5] 保存PCA结果...")
temporal_comp_path = os.path.join(output_dir, 'pca_components_temporal.npy')
frames_comp_path = os.path.join(output_dir, 'pca_components_frames.npy')
temporal_var_path = os.path.join(output_dir, 'pca_variance_temporal.npy')
frames_var_path = os.path.join(output_dir, 'pca_variance_frames.npy')

np.save(temporal_comp_path, components_temporal)
np.save(frames_comp_path, components_frames)
np.save(temporal_var_path, explained_variance_temporal)
np.save(frames_var_path, explained_variance_frames)

print(f"   pca_components_temporal.npy: {components_temporal.shape}")
print(f"   pca_components_frames.npy: {components_frames.shape}")
print(f"   pca_variance_temporal.npy: {explained_variance_temporal.shape}")
print(f"   pca_variance_frames.npy: {explained_variance_frames.shape}")
print(f"   pca_heatmap_temporal.png")
print(f"   pca_heatmap_frames.png")
print(f"   pca_scatter_temporal.png")
print(f"   pca_scatter_frames.png")
print(f"   pca_timecurve_frames.png")
print(f"   pca_analysis_log.txt")

print("\n" + "=" * 60)
print(f" PCA分析完成! 结果保存在: {output_dir}")
print("=" * 60)

# ========== 可视化函数 ==========
def visualize_pca_in_simulation(task_name_input, embodiment_name_input, config_type_input):
    """在仿真环境中可视化PCA结果

    Args:
        task_name_input: 任务名（从zarr路径提取）
        embodiment_name_input: embodiment名称（从zarr文件名提取）
        config_type_input: 配置类型 ('clean' 或 'randomized')
    """

    print("\n" + "=" * 60)
    print("开始生成仿真视频")
    print("=" * 60)

    # 使用传入的参数，不再重新解析
    task_name = task_name_input
    embodiment_name_raw = embodiment_name_input
    config_type = config_type_input

    # Embodiment名称标准化映射
    # 数据集文件名中的embodiment可能与配置文件中的名称不完全一致
    embodiment_mapping = {
        'arx-x5': 'ARX-X5',
        'arx': 'ARX-X5',
        'franka': 'franka-panda',
        'ur5': 'ur5-wsg',
        'aloha-agilex': 'aloha-agilex',
        'piper': 'piper'
    }

    # 标准化embodiment名称
    embodiment_name = embodiment_mapping.get(embodiment_name_raw.lower(), embodiment_name_raw)

    print(f"\n使用参数:")
    print(f"  任务名: {task_name}")
    print(f"  Embodiment (原始): {embodiment_name_raw}")
    if embodiment_name != embodiment_name_raw:
        print(f"  Embodiment (映射后): {embodiment_name}")
    print(f"  配置类型: {config_type}")

    # 加载任务配置
    config_path = f"/home/wangbo/RoboTwin/task_config/{args.task_config}.yml"
    if not os.path.exists(config_path):
        print(f"⚠️  配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        task_args = yaml.load(f, Loader=yaml.FullLoader)

    task_args['task_name'] = task_name

    # 设置embodiment
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f, Loader=yaml.FullLoader)

    # 检查embodiment是否在配置中
    if embodiment_name not in _embodiment_types:
        print(f"⚠️  Embodiment '{embodiment_name}' 不在配置文件中")
        print(f"    可用的embodiments: {list(_embodiment_types.keys())}")
        print(f"    跳过可视化...")
        return

    task_args['embodiment'] = [embodiment_name]
    print(f"   Embodiment配置加载成功")

    # 创建统一的视频输出目录（所有视频保存在同一个文件夹）
    # 视频命名已包含task、embodiment和config信息，无需分散到不同子目录
    video_dir = '/home/wangbo/RoboTwin/data/simulavideos'
    os.makedirs(video_dir, exist_ok=True)
    print(f"\n视频输出目录: {video_dir}")

    # ========== 可视化时序PCA ==========
    # 已注释：时序PCA视频生成
    # try:
    #     print("\n" + "-" * 60)
    #     print("可视化时序PCA - PC1")
    #     print("-" * 60)
    #     visualize_temporal_pca(task_name, embodiment_name, config_type, task_args, video_dir, args.vis_seed)
    # except Exception as e:
    #     print(f"⚠️  时序PCA可视化失败: {e}")
    #     import traceback
    #     traceback.print_exc()

    # ========== 可视化帧级PCA ==========
    try:
        print("\n" + "-" * 60)
        print("可视化帧级PCA - PC1")
        print("-" * 60)
        visualize_framewise_pca(task_name, embodiment_name, config_type, task_args, video_dir, args.vis_seed, args.framewise_steps)
    except Exception as e:
        print(f"⚠️  帧级PCA可视化失败: {e}")
        import traceback
        traceback.print_exc()

def setup_task_environment(task_name, task_args, seed, video_dir):
    """初始化任务环境
    """

    # 获取embodiment配置
    embodiment_type = task_args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f, Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f, Loader=yaml.FullLoader)
        return embodiment_args

    # ========== 核心逻辑：正确处理embodiment配置 ==========
    if len(embodiment_type) == 1:
        embodiment_name = embodiment_type[0]
        task_args["left_robot_file"] = get_embodiment_file(embodiment_name)
        task_args["right_robot_file"] = get_embodiment_file(embodiment_name)

        embodiment_config = get_embodiment_config(task_args["left_robot_file"])
        is_dual_arm = embodiment_config.get("dual_arm", True)
        task_args["dual_arm_embodied"] = is_dual_arm

        # 如果是独立双臂（dual_arm=False），需要设置embodiment_dis
        if not is_dual_arm:
            # 设置默认间距（米）
            default_distances = {
                'ARX-X5': 0.7,        # ARX-X5两臂间距
                'franka-panda': 0.8,  # Franka Panda默认间距
                'piper': 0.75,        # Piper默认间距
                'ur5-wsg': 0.8        # UR5默认间距
            }
            task_args["embodiment_dis"] = default_distances.get(embodiment_name, 0.75)
            print(f"   独立双臂模式，设置 embodiment_dis = {task_args['embodiment_dis']}m")
        else:
            # 一体式双臂不需要embodiment_dis
            print(f"   一体式双臂模式，无需 embodiment_dis")

    elif len(embodiment_type) == 3:
        # 两个不同的embodiment + 间距
        task_args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
        print(f"   异构双臂模式，embodiment_dis = {task_args['embodiment_dis']}m")
    else:
        raise ValueError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])

    # 创建任务实例
    try:
        envs_module = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(envs_module, task_name)
        task_env = env_class()
    except ModuleNotFoundError as e:
        print(f"❌ 错误: 无法加载任务模块 'envs.{task_name}'")
        print(f"    请确认任务名称正确，且对应的Python文件存在")
        print(f"    任务文件路径应为: /home/wangbo/RoboTwin/envs/{task_name}.py")
        raise e
    except AttributeError as e:
        print(f"❌ 错误: 模块 'envs.{task_name}' 中找不到类 '{task_name}'")
        print(f"    请检查任务类定义是否正确")
        raise e

    # 设置评估模式
    task_args['eval_mode'] = True
    task_args['render_freq'] = 0
    task_args['eval_video_save_dir'] = video_dir

    # 启用第三人称观察视角，确保能看到机械臂
    task_args['data_type']['third_view'] = True
    task_args['eval_video_camera_type'] = 'third_view'  # 使用observer相机录制视频
    task_args['disable_success_check'] = True  # PCA可视化时禁用成功检查，执行完整轨迹

    # 初始化环境
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **task_args)

    return task_env, task_args

# 已注释：时序PCA视频生成函数
# def visualize_temporal_pca(task_name, embodiment_name, config_type, task_args, video_dir, seed):
#     """可视化时序PCA - 直接执行PC1轨迹
#
#     Args:
#         task_name: 任务名
#         embodiment_name: embodiment名称
#         config_type: 配置类型 ('clean' 或 'randomized')
#         task_args: 任务参数
#         video_dir: 视频输出目录
#         seed: 随机种子
#     """
#
#     # 初始化环境
#     task_env, task_args = setup_task_environment(task_name, task_args, seed, video_dir)
#
#     # 获取视频配置
#     camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
#     with open(camera_config_path, "r", encoding="utf-8") as f:
#         camera_config = yaml.load(f, Loader=yaml.FullLoader)
#
#     head_camera_type = task_args["camera"]["head_camera_type"]
#     video_size = f"{camera_config[head_camera_type]['w']}x{camera_config[head_camera_type]['h']}"
#
#     #  创建视频管道（文件名包含任务、embodiment和config）
#     video_filename = f"{task_name}-{embodiment_name}-{config_type}-temporal_pca_pc1.mp4"
#     video_path = os.path.join(video_dir, video_filename)
#     ffmpeg = subprocess.Popen([
#         "ffmpeg", "-y", "-loglevel", "error",
#         "-f", "rawvideo", "-pixel_format", "rgb24",
#         "-video_size", video_size, "-framerate", "10",
#         "-i", "-", "-pix_fmt", "yuv420p",
#         "-vcodec", "libx264", "-crf", "23", video_path
#     ], stdin=subprocess.PIPE)
#
#     task_env._set_eval_video_ffmpeg(ffmpeg)
#
#
#     # 1. 计算数据中心（均值轨迹）
#     mean_trajectory = X_temporal.mean(axis=0).reshape(T_max, action_dim)
#
#     # 2. 提取PC1方向
#     PC1 = components_temporal[0].reshape(T_max, action_dim)
#
#     # 3. 选择投影最大的episode进行重建
#     Z_temporal_local = X_temporal @ components_temporal.T
#     episode_idx = np.argmax(np.abs(Z_temporal_local[:, 0]))
#     alpha = Z_temporal_local[episode_idx, 0]
#
#     # 4. 重建轨迹 = 均值 + 投影系数 × PC1
#     trajectory_norm = mean_trajectory + alpha * PC1
#
#     print(f"  轨迹长度: {T_max}步")
#     print(f"  PC1解释方差: {explained_variance_temporal[0]:.2%}")
#     print(f"  选择episode {episode_idx} (PC1投影: {alpha:.4f})")
#
#     # 反归一化
#     trajectory = normalizer.unnormalize(trajectory_norm)
#     if not isinstance(trajectory, np.ndarray):
#         trajectory = trajectory.cpu().numpy()
#
#     print(f"  反归一化完成")
#     print(f"\n  开始执行...")
#     task_env.get_obs()
#
#     # 逐步执行
#     print(f"\n  开始逐帧执行 {T_max} 步...")
#     for t in range(T_max):
#         action = trajectory[t]
#         if t % 10 == 0:
#             print(f"  步数: {t}/{T_max}", end='\r')
#
#     # 关闭视频
#     task_env._del_eval_video_ffmpeg()
#     task_env.close_env()
#
#     print(f"\n   视频已保存: {video_path}")
#     print(f"  最终步数: {task_env.take_action_cnt}")
#     print(f"  任务成功: {'是' if task_env.eval_success else '否'}")

def visualize_framewise_pca(task_name, embodiment_name, config_type, task_args, video_dir, seed, n_steps):
    """可视化帧级PCA - 沿PC1方向往复运动

    Args:
        task_name: 任务名
        embodiment_name: embodiment名称
        config_type: 配置类型 ('clean' 或 'randomized')
        task_args: 任务参数
        video_dir: 视频输出目录
        seed: 随机种子
        n_steps: 每个方向的步数
    """

    # 初始化环境
    task_env, task_args = setup_task_environment(task_name, task_args, seed, video_dir)

    # 获取视频配置
    camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_config = yaml.load(f, Loader=yaml.FullLoader)

    head_camera_type = task_args["camera"]["head_camera_type"]
    video_size = f"{camera_config[head_camera_type]['w']}x{camera_config[head_camera_type]['h']}"

    #  创建视频管道（文件名包含任务、embodiment和config）
    video_filename = f"{task_name}-{embodiment_name}-{config_type}-framewise_pca_pc1.mp4"
    video_path = os.path.join(video_dir, video_filename)
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", video_size, "-framerate", "10",
        "-i", "-", "-pix_fmt", "yuv420p",
        "-vcodec", "libx264", "-crf", "23", video_path
    ], stdin=subprocess.PIPE)

    task_env._set_eval_video_ffmpeg(ffmpeg)

    # 1. 计算归一化空间的均值action（数据中心）
    if isinstance(actions_norm, np.ndarray):
        mean_action = actions_norm.mean(axis=0).astype(np.float32)
    else: 
        mean_action = actions_norm.mean(dim=0).cpu().numpy().astype(np.float32)

    # 2. 提取PC1方向和原始方差
    pc1_direction = components_frames[0].astype(np.float32)  # (action_dim,)

    #  使用原始方差（explained_variance_）而不是方差解释率
    explained_var_raw = explained_variance_frames_raw  # 从全局加载
    sigma = float(np.sqrt(explained_var_raw[0]))

    print(f"  PC1解释方差比率: {explained_variance_frames[0]:.2%}")
    print(f"  PC1原始方差: {explained_var_raw[0]:.6f}")
    print(f"  标准差σ (从原始方差): {sigma:.4f}")
    print(f"  运动范围: -3σ ({-3*sigma:.3f}) 到 +3σ ({3*sigma:.3f})")

    # 3. 生成轨迹: 均值 + (-3σ → +3σ → -3σ) × PC1方向
    alphas_forward = np.linspace(-3*sigma, 3*sigma, n_steps)
    alphas_backward = np.linspace(3*sigma, -3*sigma, n_steps)
    alphas = np.concatenate([alphas_forward, alphas_backward])

    # 使用广播和矩阵运算，避免列表推导式
    trajectory_norm = mean_action[np.newaxis, :] + alphas[:, np.newaxis] * pc1_direction[np.newaxis, :]
    trajectory_norm = trajectory_norm.astype(np.float32)

    print(f"  轨迹长度: {len(trajectory_norm)}步 (往复)")

    # 反归一化
    trajectory = normalizer.unnormalize(trajectory_norm)
    if not isinstance(trajectory, np.ndarray):
        trajectory = trajectory.cpu().numpy()
    print(f"\n  开始执行...")

    #初始化观测，确保now_obs被填充
    task_env.get_obs()

    # 逐步执行
    for t in range(len(trajectory)):
        action = trajectory[t]
        task_env.take_action(action, action_type='qpos')
        if t % 10 == 0:
            direction = "前进" if t < n_steps else "后退"
            print(f"  步数: {t}/{len(trajectory)} ({direction})", end='\r')

    # 关闭视频
    task_env._del_eval_video_ffmpeg()
    task_env.close_env()

    print(f"\n   视频已保存: {video_path}")
    print(f"  最终步数: {task_env.take_action_cnt}")
    print(f"  任务成功: {'是' if task_env.eval_success else '否'}")

# ========== 执行可视化 ==========
if args.visualize:
    # 从zarr文件名中提取embodiment名称和config类型
    # zarr_filename格式: {embodiment}_{config}_{episodes}
    # 例如: arx-x5_randomized_500, franka_clean_50

    if task_name_extracted is None:
        print("\n⚠️  跳过可视化：无法从路径中提取任务名")
        print(f"    请确保zarr文件位于标准路径: /path/to/{{task_name}}/{{embodiment}}_{{config}}_{{episodes}}.zarr")
    else:
        # 从zarr_filename提取embodiment和config
        if '_clean_' in zarr_filename:
            embodiment_name = zarr_filename.split('_clean_')[0]
            config_type = 'clean'
        elif '_randomized_' in zarr_filename:
            embodiment_name = zarr_filename.split('_randomized_')[0]
            config_type = 'randomized'
        else:
            print("\n⚠️  跳过可视化：无法解析embodiment名称和配置类型")
            print(f"    zarr文件名: {zarr_filename}")
            embodiment_name = None
            config_type = None

        if embodiment_name and config_type:
            visualize_pca_in_simulation(task_name_extracted, embodiment_name, config_type)

print("\n" + "=" * 60)
print(" 全部任务完成!")
print("=" * 60)

# 关闭日志文件
logger.close()
