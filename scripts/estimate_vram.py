"""Estimate VRAM usage for different model configs on RTX 5090."""
import torch
import sys
sys.path.insert(0, ".")
from src.nilmformer.config import NILMFormerConfig
from src.nilmformer.model import NILMFormer

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# c_in=1 (load curve only), c_embedding=8 (sin/cos encoded exo)
# 4 exo vars * 2 (sin+cos) = 8 channels + 1 power = 9 total
N_INPUT_CHANNELS = 9

# Current config: d_model=128, 4 layers
cfg128 = NILMFormerConfig(
    c_in=1, c_out=5, c_embedding=8,
    kernel_size=3, kernel_size_head=1,
    dilations=[1,2,4,8], conv_bias=True,
    use_efficient_attention=False,
    n_encoder_layers=4, d_model=128,
    dp_rate=0.1526, pffn_ratio=5,
    n_head=4, norm_eps=1e-5,
    kettle_channel_idx=2,
)
model_128 = NILMFormer(cfg128)
p128 = count_params(model_128)
print(f"d_model=128, 4L: {p128:,} params ({p128*4/1024/1024:.1f}MB fp32)")

# Larger: d_model=256, 6 layers
cfg256 = NILMFormerConfig(
    c_in=1, c_out=5, c_embedding=8,
    kernel_size=3, kernel_size_head=1,
    dilations=[1,2,4,8], conv_bias=True,
    use_efficient_attention=False,
    n_encoder_layers=6, d_model=256,
    dp_rate=0.15, pffn_ratio=4,
    n_head=8, norm_eps=1e-5,
    kettle_channel_idx=2,
)
model_256 = NILMFormer(cfg256)
p256 = count_params(model_256)
print(f"d_model=256, 6L: {p256:,} params ({p256*4/1024/1024:.1f}MB fp32)")

# Actual VRAM test with different batch sizes
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    for cfg_name, model, bs_list in [
        ("d128/4L", model_128, [256, 512, 1024, 2048, 4096, 8192]),
        ("d256/6L", model_256, [256, 512, 1024, 2048, 4096]),
    ]:
        model = model.to(device).to(torch.bfloat16)
        model.train()
        for bs in bs_list:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            try:
                x = torch.randn(bs, N_INPUT_CHANNELS, 128, device=device, dtype=torch.bfloat16)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(x)
                    # Simulate PCGrad: 5 backward passes with retain_graph
                    for i in range(5):
                        loss_i = y[:, i:i+1, :].sum()
                        loss_i.backward(retain_graph=(i < 4))
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                print(f"{cfg_name} BS={bs:>4d}: peak={peak_mb:.0f}MB ({peak_mb/1024:.1f}GB)")
                del x, y, loss_i
                model.zero_grad()
            except torch.cuda.OutOfMemoryError:
                print(f"{cfg_name} BS={bs:>4d}: OOM!")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"{cfg_name} BS={bs:>4d}: ERROR {e}")
                break
            finally:
                torch.cuda.empty_cache()
        model = model.cpu()
        torch.cuda.empty_cache()
