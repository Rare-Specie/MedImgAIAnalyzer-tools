import torch

def check_torch_device():
    """
    检查PyTorch使用的计算设备（CPU或具体加速平台）
    """
    print("=" * 50)
    print("PyTorch 设备检测结果")
    print("=" * 50)
    print(f"PyTorch 版本: {torch.__version__}")
    print("-" * 50)
    
    # 1. 检测 CPU（始终可用）
    cpu_available = True
    print(f"✅ CPU: 可用（所有环境默认支持）")
    
    # 2. 检测 NVIDIA CUDA（最主流的GPU加速平台，支持Windows/Linux）
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        current_cuda_device = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(current_cuda_device)
        cuda_version = torch.version.cuda
        print(f"✅ CUDA: 可用")
        print(f"   - CUDA 版本: {cuda_version}")
        print(f"   - 可用 GPU 数量: {cuda_device_count}")
        print(f"   - 当前默认 GPU: {current_cuda_device}（名称：{cuda_device_name}）")
    else:
        print(f"❌ CUDA: 不可用（未安装对应CUDA版本、无NVIDIA显卡或驱动不兼容）")
    
    # 3. 检测 Apple MPS（苹果Silicon芯片（M1/M2/M3）的加速平台）
    try:
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
    except AttributeError:
        # 低版本PyTorch不支持MPS，直接标记为不可用
        mps_available = False
        mps_built = False
    
    if mps_available and mps_built:
        print(f"✅ MPS: 可用（Apple Silicon 芯片硬件加速）")
    else:
        reason = []
        if not mps_built:
            reason.append("PyTorch 编译时未启用 MPS 支持")
        if not mps_available:
            reason.append("非 Apple Silicon 芯片或系统版本过低（需macOS 12.3+）")
        print(f"❌ MPS: 不可用（{'; '.join(reason)}）")
    
    # 4. 检测其他小众加速平台（可选，按需参考）
    # 检测 Google TPU
    try:
        tpu_available = torch.backends.xla.is_available()
    except AttributeError:
        tpu_available = False
    print(f"{'✅' if tpu_available else '❌'} TPU (XLA): {'可用（需在Google Colab/TPU环境中）' if tpu_available else '不可用'}")
    
    # 检测 Graphcore IPU
    try:
        ipu_available = torch.backends.ipu.is_available()
    except AttributeError:
        ipu_available = False
    print(f"{'✅' if ipu_available else '❌'} IPU: {'可用（Graphcore 硬件环境）' if ipu_available else '不可用'}")
    
    print("-" * 50)
    # 5. 输出 PyTorch 默认使用的设备
    if cuda_available:
        default_device = torch.device("cuda")
    elif mps_available and mps_built:
        default_device = torch.device("mps")
    else:
        default_device = torch.device("cpu")
    
    print(f"📌 PyTorch 当前默认计算设备: {default_device}")
    print("=" * 50)

if __name__ == "__main__":
    check_torch_device()