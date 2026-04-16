#!/bin/bash

# 用法说明
usage() {
    echo "用法: $0 [-a ATTENTION_DEVICES] [-f FFN_DEVICES] [-h]"
    echo "  参数:"
    echo "    -a ATTENTION_DEVICES  attention服务器使用的卡号（默认: 2,3）"
    echo "    -f FFN_DEVICES        ffn服务器使用的卡号（默认: 0,1）"
    echo "    -h                    显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -a 0,1 -f 2,3        使用卡0,1运行attention，卡2,3运行ffn"
    exit 1
}

# 默认值，指定使用attn 和 ffn卡rank
ATTENTION_DEVICES="2,3"
FFN_DEVICES="4,5"

# 解析命令行参数
while getopts "a:f:h" opt; do
    case $opt in
        a) ATTENTION_DEVICES="$OPTARG" ;;
        f) FFN_DEVICES="$OPTARG" ;;
        h) usage ;;
        \?) echo "无效选项: -$OPTARG" >&2; usage ;;
        :) echo "选项 -$OPTARG 需要参数" >&2; usage ;;
    esac
done

# 检查卡号是否冲突
check_device_conflict() {
    local dev1_arr=(${1//,/ })
    local dev2_arr=(${2//,/ })

    for dev1 in "${dev1_arr[@]}"; do
        for dev2 in "${dev2_arr[@]}"; do
            if [ "$dev1" = "$dev2" ]; then
                echo "错误: attention和ffn服务器使用了相同的卡号: $dev1"
                echo "请确保两个服务器使用不同的卡号"
                exit 1
            fi
        done
    done
}

# 检查卡号冲突
check_device_conflict "$ATTENTION_DEVICES" "$FFN_DEVICES"

echo "配置:"
echo "  Attention服务器卡号: $ATTENTION_DEVICES"
echo "  FFN服务器卡号: $FFN_DEVICES"
echo ""

# 设置公共环境变量
export HCCL_BUFFSIZE=4096
export VLLM_LOGGING_LEVEL=INFO

# 日志文件路径
LOG_DIR="/home/wjc/workspace-afd/test-vllm-ascend"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ATTENTION_LOG="$LOG_DIR/attn_${ATTENTION_DEVICES//,/_}_${TIMESTAMP}.log"
FFN_LOG="$LOG_DIR/ffn_${FFN_DEVICES//,/_}_${TIMESTAMP}.log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 启动attention服务器
echo "=== 启动attention服务器（卡号: $ATTENTION_DEVICES）==="
export ASCEND_RT_VISIBLE_DEVICES="$ATTENTION_DEVICES"
nohup vllm serve /mnt/weight/DeepSeek-V3 \
    --data-parallel-size 2 \
    --max_num_batched_tokens 20 \
    --max_num_seqs 20 \
    --enforce_eager \
    --port 8004 \
    --max-model-len 4096 \
    --quantization ascend \
    --afd-config '{"afd_connector":"camp2pconnector", "afd_role": "attention", "num_afd_stages":"2","afd_extra_config":{"afd_size":"2A2F"}, "compute_gate_on_attention": "True","afd_port":"29667", "enable_profiling": "True"}' \
    > "$ATTENTION_LOG" 2>&1 &
ATTENTION_PID=$!
# --enable-dbo \ 开启dbo
# --dbo-prefill-token-threshold 12  \dbo-prefill 阈值，超过这个阈值才切分
#--dbo-decode-token-threshold 2 \ dbo-decode 阈值，超过这个阈值才切分
# --ubatch-size 2\ # ubatch切分的个数，如果只传enable-dbo，ubatch-size=2
# afd-config ：afd_connector 指定connector类型，支持camm2nconnector、camp2pconnector
# afd_role：角色是A还是F；num_afd_stages：开启dbo才需要，表示有两个microbatch；"afd_size":"2A2F"表示用了2张卡跑A两张卡跑F；compute_gate_on_attention，是否需要在affn侧计算gating；afd_port：AF建立通信域需要的port；afd_host：AF建立通信域需要的ip
# 注意: export VLLM_TORCH_PROFILER_DIR="./vllm_profile"--指定存储路径

# 记录PID
echo "attention服务器PID: $ATTENTION_PID"
echo "日志文件: $ATTENTION_LOG"
# 启动ffn服务器
echo ""
echo "=== 启动ffn服务器（卡号: $FFN_DEVICES）==="
export ASCEND_RT_VISIBLE_DEVICES="$FFN_DEVICES"
nohup python -m vllm.entrypoints.afd_ffn_server /mnt/weight/DeepSeek-V3 \
    --tensor-parallel-size 2 \
    --enable_expert_parallel \
    --max_num_batched_tokens 20 \
    --enforce_eager \
    --max_num_seqs 20 \
    --max-model-len 4096 \
    --quantization ascend \
    --afd-config '{"afd_connector":"camp2pconnector", "num_afd_stages":"2", "afd_role": "ffn", "afd_extra_config":{"afd_size":"2A2F"}, "compute_gate_on_attention": "True","afd_port":"29667", "enable_profiling": "True"}' \
    > "$FFN_LOG" 2>&1 &
FFN_PID=$!
# 记录PID
echo "ffn服务器PID: $FFN_PID"
echo "日志文件: $FFN_LOG"

# 保存PID到文件，方便后续管理
PIDS_FILE="$LOG_DIR/server_pids_$(date +%Y%m%d_%H%M%S).txt"
echo "ATTENTION_PID=$ATTENTION_PID" > "$PIDS_FILE"
echo "FFN_PID=$FFN_PID" >> "$PIDS_FILE"
echo "ATTENTION_LOG=$ATTENTION_LOG" >> "$PIDS_FILE"
echo "FFN_LOG=$FFN_LOG" >> "$PIDS_FILE"
echo "ATTENTION_DEVICES=$ATTENTION_DEVICES" >> "$PIDS_FILE"
echo "FFN_DEVICES=$FFN_DEVICES" >> "$PIDS_FILE"

echo ""
echo "=== 服务器启动完成 ==="
echo "两个服务器已同时拉起并运行在后台"
echo "PID文件: $PIDS_FILE"
echo ""
echo "监控日志:"
echo "  tail -f $ATTENTION_LOG"
echo "  tail -f $FFN_LOG"
echo ""
echo "停止服务器:"
echo "  kill $ATTENTION_PID $FFN_PID"
echo "或使用停止脚本: $0 stop (如果实现了停止功能)"

# 提供一个简单的监控选项
read -p "是否打开实时日志监控? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "=== 实时日志监控（Ctrl+C退出监控，服务器继续运行）==="
    echo "注意: 按 Ctrl+C 只退出监控，服务器继续在后台运行"
    echo ""
    # 检查是否安装了multitail
    if command -v multitail &> /dev/null; then
        multitail -s 2 \
            -l "tail -f $ATTENTION_LOG" \
            -l "tail -f $FFN_LOG" \
            -cs attention -T "Attention Server ($ATTENTION_DEVICES)" \
            -cs ffn -T "FFN Server ($FFN_DEVICES)"
    else
        echo "ATTENTION 服务器日志 ($ATTENTION_DEVICES):"
        echo "FFN 服务器日志 ($FFN_DEVICES):"
        echo "----------------------------------------"
        # 使用简单的tail -f同时查看两个日志
        (tail -f "$ATTENTION_LOG" & tail -f "$FFN_LOG") | awk '/^==>/ {print "\033[32m" $0 "\033[0m"; next} {print}'
    fi
fi