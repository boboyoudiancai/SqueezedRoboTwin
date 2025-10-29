#!/bin/bash
# ==================== 配置参数 ====================
# ✓ 自动扫描模式：不再硬编码任务名，自动处理所有数据集
DATASET_ROOT="/home/wangbo/data/dataset"  # 数据集根目录
# zarr文件直接保存在各任务目录下，例如: /home/wangbo/data/dataset/{task_name}/{config}_{num}.zarr
SCRIPT_DIR="/home/wangbo/RoboTwin/policy/DP"
NUM_EPISODES=50

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

# ==================== 函数定义 ====================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# 检查目录是否存在
check_directories() {
    print_header "检查目录"

    if [ ! -d "$DATASET_ROOT" ]; then
        print_error "数据集根目录不存在: $DATASET_ROOT"
        exit 1
    fi
    print_success "数据集根目录: $DATASET_ROOT"

    if [ ! -d "$SCRIPT_DIR" ]; then
        print_error "脚本目录不存在: $SCRIPT_DIR"
        exit 1
    fi
    print_success "脚本目录: $SCRIPT_DIR"

    echo ""
}

# ✓ 扫描所有任务目录，获取所有 zip 文件
scan_all_datasets() {
    cd "$DATASET_ROOT"
    # 扫描所有任务目录下的 .zip 文件（深度限制为2层）
    find . -maxdepth 2 -name "*.zip" -type f | sort
}

# ✓ 检查解压文件完整性
check_extraction_integrity() {
    local target_dir=$1
    local expected_episodes=$2

    # 检查 data 目录是否存在
    if [ ! -d "$target_dir/data" ]; then
        return 1
    fi

    # 统计 HDF5 文件数量
    local num_hdf5=$(find "$target_dir/data" -name "*.hdf5" 2>/dev/null | wc -l)

    # 检查文件数量是否匹配（允许一定误差，因为可能有失败的episode）
    # 至少要有 expected_episodes 的 80%
    local min_required=$((expected_episodes * 80 / 100))

    if [ $num_hdf5 -ge $min_required ]; then
        return 0
    else
        return 1
    fi
}

# ✓ 解压单个 zip 文件（自动检测目标目录，检查完整性）
unzip_file() {
    local zip_file=$1  # 相对路径，例如: ./handover_mic/aloha-agilex_clean_50.zip
    local expected_episodes=$2  # 期望的 episode 数量

    local base_name=$(basename "$zip_file" .zip)
    local dir_name=$(dirname "$zip_file")

    # 转换为绝对路径（规范化，去掉./）
    local abs_zip_file="$DATASET_ROOT/${zip_file#./}"
    local abs_extract_dir="$DATASET_ROOT/${dir_name#./}"
    local target_dir="$abs_extract_dir/$base_name"

    # ✓ 检查是否已解压且完整
    if [ -d "$target_dir" ]; then
        if check_extraction_integrity "$target_dir" "$expected_episodes"; then
            print_info "已解压且完整，跳过: $base_name"
            return 0
        else
            print_error "解压文件不完整，删除并重新解压: $base_name"
            rm -rf "$target_dir"
        fi
    fi

    print_info "解压: $base_name"

    # 解压
    cd "$abs_extract_dir"
    unzip -q "$abs_zip_file" -d "$abs_extract_dir"

    if [ $? -eq 0 ]; then
        # 验证解压完整性
        if check_extraction_integrity "$target_dir" "$expected_episodes"; then
            print_success "解压完成且完整: $base_name"
            return 0
        else
            print_error "解压完成但文件不完整: $base_name"
            return 1
        fi
    else
        print_error "解压失败: $abs_zip_file"
        return 1
    fi
}

# ✓ 处理单个数据集（自动提取任务名）
process_dataset() {
    local task_name=$1     # 例如: handover_mic
    local config_name=$2   # 例如: aloha-agilex_clean
    local num_episodes=$3  # 例如: 50

    print_info "处理数据集: ${task_name} - ${config_name}"

    cd "$SCRIPT_DIR"

    # 运行 policy/DP/process_data.sh
    bash process_data.sh "$task_name" "$config_name" "$num_episodes"

    if [ $? -eq 0 ]; then
        print_success "数据处理完成: ${task_name} - ${config_name}"

        # 验证输出文件（新路径：保存在dataset目录下）
        local output_file="${DATASET_ROOT}/${task_name}/${config_name}_${num_episodes}.zarr"
        if [ -d "$output_file" ]; then
            local size=$(du -sh "$output_file" | cut -f1)
            print_success "输出文件: $output_file (大小: $size)"
            return 0
        else
            print_error "输出文件不存在: $output_file"
            return 1
        fi
    else
        print_error "数据处理失败: ${task_name} - ${config_name}"
        return 1
    fi
}

# ✓ 从文件名提取信息
# 格式: {embodiment}_{config}_{episodes}.zip
# 例如: aloha-agilex_clean_50.zip -> embodiment=aloha-agilex, config=clean, episodes=50
parse_filename() {
    local filename=$1
    local base_name=$(basename "$filename" .zip)

    # 提取 episodes（最后的数字）
    local episodes=$(echo "$base_name" | grep -oE '[0-9]+$')

    # 移除 episodes 部分，剩余: {embodiment}_{config}_
    local without_episodes="${base_name%_${episodes}}"

    # 提取 config（最后一个下划线后的部分）
    local config=$(echo "$without_episodes" | grep -oE '[^_]+$')

    # 提取 embodiment（剩余部分）
    local embodiment="${without_episodes%_${config}}"

    echo "$embodiment|$config|$episodes"
}

# 清理解压文件（可选）
cleanup_extracted() {
    local dir_path=$1

    if [ "$CLEANUP" = "true" ]; then
        print_info "清理解压文件: $(basename "$dir_path")"
        rm -rf "$dir_path"
        print_success "清理完成"
    fi
}

# ==================== 主流程 ====================

main() {
    print_header "🚀 自动扫描并处理所有数据集"

    # 检查环境
    check_directories

    # ✓ 扫描所有 zip 文件
    print_header "扫描数据集"
    cd "$DATASET_ROOT"

    # 使用 mapfile 读取所有 zip 文件
    mapfile -t zip_files < <(find . -maxdepth 2 -name "*.zip" -type f | sort)

    if [ ${#zip_files[@]} -eq 0 ]; then
        print_error "未找到任何 .zip 文件"
        exit 1
    fi

    print_success "找到 ${#zip_files[@]} 个压缩文件:"
    for zip_file in "${zip_files[@]}"; do
        echo "  - $zip_file"
    done
    echo ""

    # 统计变量
    total=${#zip_files[@]}
    success=0
    failed=0
    skipped=0

    # ✓ 处理每个数据集
    for i in "${!zip_files[@]}"; do
        zip_file="${zip_files[$i]}"
        base_name=$(basename "$zip_file" .zip)
        dir_name=$(dirname "$zip_file")

        # 提取任务名（从目录名，例如 ./handover_mic/xxx.zip -> handover_mic）
        task_name=$(basename "$dir_name")

        # 解析文件名获取配置信息
        IFS='|' read -r embodiment config episodes <<< "$(parse_filename "$zip_file")"

        print_header "[$((i+1))/$total] 处理: $task_name/$base_name"
        print_info "任务: $task_name | Embodiment: $embodiment | Config: $config | Episodes: $episodes"

        # 步骤 1: 解压（并检查完整性）
        if ! unzip_file "$zip_file" "$episodes"; then
            print_error "跳过处理: $base_name"
            ((failed++))
            continue
        fi

        # 步骤 2: 验证解压内容
        local extracted_dir="$DATASET_ROOT/${dir_name#./}/$base_name"
        if [ ! -d "$extracted_dir/data" ]; then
            print_error "数据目录不存在: $extracted_dir/data"
            ((failed++))
            continue
        fi

        local num_hdf5=$(find "$extracted_dir/data" -name "*.hdf5" 2>/dev/null | wc -l)
        print_info "找到 $num_hdf5 个 HDF5 文件 (期望: $episodes, 最低要求: $((episodes * 80 / 100)))"

        # 步骤 3: 构建配置名称
        local config_name="${embodiment}_${config}"

        # 检查输出文件是否已存在（新路径：保存在dataset目录下）
        local output_file="${DATASET_ROOT}/${task_name}/${config_name}_${episodes}.zarr"
        if [ -d "$output_file" ]; then
            print_info "输出文件已存在，跳过处理: $(basename "$output_file")"
            ((skipped++))

            # 可选：清理解压文件
            if [ "$CLEANUP" = "true" ]; then
                cleanup_extracted "$extracted_dir"
            fi

            echo ""
            continue
        fi

        # 步骤 4: 处理数据
        if process_dataset "$task_name" "$config_name" "$episodes"; then
            ((success++))

            # 可选：清理解压文件以节省空间
            if [ "$CLEANUP" = "true" ]; then
                cleanup_extracted "$extracted_dir"
            fi
        else
            ((failed++))
        fi

        echo ""
    done

    # ==================== 最终报告 ====================
    print_header "📊 处理完成"
    echo -e "${GREEN}✓ 成功: $success${NC}"
    echo -e "${YELLOW}⊘ 跳过: $skipped${NC}"
    echo -e "${RED}✗ 失败: $failed${NC}"
    echo -e "━━━━━━━━━━━━━━━━━━"
    echo -e "总计: $total"
    echo ""

    if [ $success -gt 0 ] || [ $skipped -gt 0 ]; then
        print_header "生成的 Zarr 文件"
        # 在dataset目录下的各个任务目录中查找zarr文件
        find "$DATASET_ROOT" -maxdepth 2 -name "*.zarr" -type d | while read -r zarr_file; do
            local size=$(du -sh "$zarr_file" | cut -f1)
            echo "  - $(basename "$zarr_file") ($size)"
        done
    fi

    if [ $failed -eq 0 ]; then
        print_success "所有数据集处理成功！"
        exit 0
    else
        print_error "部分数据集处理失败 ($failed/$total)"
        exit 1
    fi
}

# ==================== 脚本入口 ====================

# 显示帮助信息
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "✓ 自动扫描并处理数据集根目录下的所有 .zip 文件"
    echo "✓ 智能跳过已解压的文件和已处理的数据集"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  --cleanup      处理完成后删除解压文件（节省空间）"
    echo ""
    echo "目录配置:"
    echo "  数据集根目录: $DATASET_ROOT"
    echo "  zarr输出位置: 各任务目录下 (例如: {task_name}/{config}_{num}.zarr)"
    echo "  处理脚本: $SCRIPT_DIR/process_data.sh"
    echo ""
    echo "功能:"
    echo "  - 自动扫描 $DATASET_ROOT 下所有任务目录"
    echo "  - 解压所有 .zip 文件（已解压的自动跳过）"
    echo "  - 自动提取任务名、embodiment、配置名和episode数"
    echo "  - 处理数据并生成 .zarr 文件"
    echo "  - 已存在的 .zarr 文件自动跳过"
    echo ""
    exit 0
fi

# 检查是否启用清理
CLEANUP=false
if [ "$1" = "--cleanup" ]; then
    CLEANUP=true
    print_info "启用自动清理模式"
fi

# 执行主流程
main
