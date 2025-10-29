#!/bin/bash
# ========== 颜色输出 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ========== 配置参数 ==========
# 数据目录
DATA_DIR="/home/wangbo/data/dataset"

# PCA脚本路径
PCA_SCRIPT="/home/wangbo/RoboTwin/pca/pca_analysis.py"

# 输出目录
OUTPUT_DIR="/home/wangbo/RoboTwin/pca/results"

# Python解释器（激活conda环境）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
PYTHON_CMD="python"

# 并行处理数量（同时运行的PCA任务数量，建议根据CPU核心数调整）
MAX_PARALLEL=10  # 默认串行处理，避免内存问题

# ========== 检查依赖 ==========
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}RoboTwin PCA 自动化分析脚本${NC}"
echo -e "${CYAN}============================================================${NC}"

echo -e "\n${BLUE}[检查] 检查依赖...${NC}"

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_DIR${NC}"
    exit 1
fi

# 检查PCA脚本
if [ ! -f "$PCA_SCRIPT" ]; then
    echo -e "${RED}错误: PCA脚本不存在: $PCA_SCRIPT${NC}"
    exit 1
fi

# 检查Python环境
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}错误: Python未找到: $PYTHON_CMD${NC}"
    exit 1
fi

# 检查Python依赖包
echo -e "${BLUE}[检查] 检查Python依赖包...${NC}"
required_packages=("zarr" "numpy" "sklearn" "matplotlib")
missing_packages=()

for pkg in "${required_packages[@]}"; do
    if ! $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        missing_packages+=($pkg)
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "${RED}错误: 缺少以下Python包: ${missing_packages[*]}${NC}"
    echo -e "${YELLOW}请运行: pip install zarr numpy scikit-learn matplotlib${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 所有依赖检查通过${NC}"

# ========== 查找所有zarr数据集 ==========
echo -e "\n${BLUE}[扫描] 查找所有zarr数据集...${NC}"

# 使用数组存储数据集路径
zarr_datasets=()
while IFS= read -r -d '' zarr_path; do
    zarr_datasets+=("$zarr_path")
done < <(find "$DATA_DIR" -name "*.zarr" -type d -print0 | sort -z)

# 显示找到的数据集
total_datasets=${#zarr_datasets[@]}
echo -e "${GREEN}✓ 找到 $total_datasets 个数据集:${NC}"
for i in "${!zarr_datasets[@]}"; do
    dataset_name=$(basename "${zarr_datasets[$i]}")
    echo -e "  ${CYAN}[$((i+1))/$total_datasets]${NC} $dataset_name"
done

if [ $total_datasets -eq 0 ]; then
    echo -e "${YELLOW}警告: 未找到任何zarr数据集${NC}"
    exit 0
fi

# ========== 创建输出目录 ==========
mkdir -p "$OUTPUT_DIR"

# ========== 执行PCA分析 ==========
echo -e "\n${CYAN}============================================================${NC}"
echo -e "${CYAN}开始PCA分析${NC}"
echo -e "${CYAN}============================================================${NC}"

# 统计变量
success_count=0
failed_count=0
skipped_count=0
failed_datasets=()

# 开始时间
start_time=$(date +%s)

# 遍历所有数据集
for i in "${!zarr_datasets[@]}"; do
    zarr_path="${zarr_datasets[$i]}"
    dataset_name=$(basename "$zarr_path" .zarr)

    echo -e "\n${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}[$(($i+1))/$total_datasets] 处理: ${CYAN}$dataset_name${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"

    # 检查输出目录是否已存在结果
    result_dir="$OUTPUT_DIR/$dataset_name"
    log_file="$result_dir/pca_analysis_log.txt"

    # 检查是否已经分析过（可选：添加 --force 参数强制重新分析）
    if [ -f "$log_file" ]; then
        echo -e "${YELLOW}⚠ 已存在分析结果，跳过...${NC}"
        echo -e "${YELLOW}  (删除 $result_dir 可重新分析)${NC}"
        ((skipped_count++))
        continue
    fi

    # 运行PCA分析（带可视化）
    echo -e "${BLUE}运行PCA分析...${NC}"

    # 切换到RoboTwin根目录，避免路径问题
    cd /home/wangbo/RoboTwin

    if $PYTHON_CMD "$PCA_SCRIPT" "$zarr_path" --output_dir "$OUTPUT_DIR" --visualize --task_config demo_clean --vis_seed 0 --framewise_steps 20; then
        echo -e "${GREEN}✓ 成功完成: $dataset_name${NC}"
        ((success_count++))
    else
        echo -e "${RED}✗ 失败: $dataset_name${NC}"
        failed_datasets+=("$dataset_name")
        ((failed_count++))
    fi

    # 显示进度
    echo -e "${CYAN}进度: 成功=$success_count, 失败=$failed_count, 跳过=$skipped_count${NC}"
done

# ========== 生成总结报告 ==========
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo -e "\n${CYAN}============================================================${NC}"
echo -e "${CYAN}PCA分析完成总结${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}✓ 成功: $success_count 个数据集${NC}"
echo -e "${YELLOW}⚠ 跳过: $skipped_count 个数据集${NC}"
echo -e "${RED}✗ 失败: $failed_count 个数据集${NC}"
echo -e "${BLUE}总耗时: ${elapsed_time}秒 ($(($elapsed_time / 60))分钟)${NC}"
echo -e "${BLUE}结果目录: $OUTPUT_DIR${NC}"

# 显示失败的数据集
if [ $failed_count -gt 0 ]; then
    echo -e "\n${RED}失败的数据集:${NC}"
    for dataset in "${failed_datasets[@]}"; do
        echo -e "  ${RED}✗${NC} $dataset"
    done
fi

echo -e "\n${CYAN}============================================================${NC}"
echo -e "${GREEN}所有任务完成！${NC}"
echo -e "${CYAN}============================================================${NC}"

# 退出状态码
if [ $failed_count -gt 0 ]; then
    exit 1
else
    exit 0
fi
