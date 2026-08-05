# MiniMax-H3 原生音视频任务

该集成支持 `t2av`、`i2av`、`l2av`、`fl2av` 和 `ref2av`。运行时不导入 Diffusers，也不需要转换或重新下载权重；DiT、Qwen3-VL 前 50 层与视觉塔、视频/音频 VAE 和 scheduler 都由 LightX2V 原生执行。Transformers 只用于 tokenizer 与官方 Qwen3-VL 像素 processor。

## 模型目录

推荐继续使用 `/data/nvme6/gushiqiao/models/MiniMax-H3` 作为 `model_path`。根目录至少需要以下组件：

```text
MiniMax-H3/
├── transformer/
├── transformer_ref/
├── vae/
├── audio_vae/
├── text_encoder/
├── tokenizer/
└── processor/
```

`text_encoder`、`tokenizer` 和 `processor` 可以是指向 `FL2VA/` 对应目录的软链接，不需要复制权重。

基础四任务读取 `transformer/`；`ref2av` 自动读取结构相同但数值不同的 `transformer_ref/`。

原生 loader 保留发布权重中的混合精度：主体权重为 BF16，原有的 FP32 投影、时间嵌入和输出头保持 FP32。运行时要求 `DTYPE=BF16`；不要求设置 `SENSITIVE_LAYER_DTYPE`，随附脚本设置它只是为了显式表达精度策略。

## 任务与输入

- `t2av`：仅 prompt。
- `i2av`：`--image_path`，首帧条件。
- `l2av`：`--last_frame_path`，尾帧条件。
- `fl2av`：同时传首帧与尾帧。
- `ref2av`：重复传 `--reference image=/path`、`video=/path` 或 `audio=/path`，顺序会被保留。最多 9 图、3 视频、3 个带音频 reference、总计 12 项；不能只有音频。

- 当前支持单 prompt、batch size 1。
- 固定 24 fps，官方时长范围为 5–15 秒。
- 帧数必须满足 `17 * n + 5`；默认是 124 帧。不满足时会先向上对齐，再检查是否仍在时长范围内。
- 高和宽都必须是 32 的倍数；默认分辨率为 768×1344。
- 模型是 guidance-distilled，不使用 CFG，`negative_prompt` 会被忽略。
- 保存路径必须以 `.mp4` 结尾；输出为 24 fps H.264 视频和 32 kHz AAC 立体声音频。
- 默认 attention 后端是 `torch_sdpa`。只有环境已经安装并支持 FlashAttention 3 时，才应把配置中的 `attn_type` 改为 `flash_attn3`。

当前实现要求按组件顺序执行 CPU offload：文本编码器 → DiT → 视频 VAE → 音频 VAE。配置中应保持 `cpu_offload`、`text_encoder_cpu_offload` 和 `vae_cpu_offload` 为 `true`。H3 支持两种 DiT offload 粒度：

- `module`：整个 DiT 在去噪开始前搬到设备，结束后搬回 CPU；旧配置值 `model` 作为兼容别名保留。
- `block`：50 个 transformer blocks 常驻 pinned CPU memory，设备上保留两个 block buffer，计算当前 block 时异步预取下一个 block。

纯文本 `t2av` 还支持独立的 Qwen3-VL tensor parallel。`parallel.tensor_p_size`
只控制 DiT，`parallel.qwen3vl_tensor_p_size` 只控制 Qwen3-VL；后者是阶段性的
进程子组，不参与 distributed world size 的乘积。当前支持能同时整除 64 个 Q heads、
8 个 KV heads、25600 intermediate size 和 151936 vocabulary size 的值，在 8 卡上
通常使用 1、2、4 或 8。省略时默认为 1，以保持原来的单 `aux_rank` 编码行为。

8 卡 DiT TP=8、Qwen3-VL TP=8 + block offload 使用：

```bash
MODEL_PATH=/llm/models/MiniMax-H3/FL2VA \
bash scripts/minimax_h3/run_minimax_h3_t2av_tp8.sh
```

Qwen TP 目前仅覆盖无图像输入的 `t2av`；视觉塔仍保持单 rank 路径。

DiT 支持 Ulysses sequence parallel。H3 的 video rows 在 SP ranks 间切分，变长的 text/audio rows 在组内复制，因此不会为 packed sequence 引入会污染 softmax 的 padding token。当前只支持 `seq_p_attn_type: "ulysses"`，且 `seq_p_size` 必须整除 56 个 attention heads（例如 2、4、7、8）。默认使用 4 卡：

```bash
MODEL_PATH=/llm/models/MiniMax-H3/FL2VA \
bash scripts/minimax_h3/run_minimax_h3_t2av_ulysses.sh
```

脚本默认使用 block offload 配置 `configs/minimax_h3/minimax_h3_t2av_ulysses_block_offload.json`。如需修改卡数，必须同时修改配置中的 `parallel.seq_p_size` 和脚本的 `NUM_PROCESSES`。若要改用 module offload，可通过 `CONFIG_JSON=configs/minimax_h3/minimax_h3_t2av_ulysses.json` 显式指定。

8 卡 TP=2 + Ulysses SP=4 + block offload 使用：

```bash
MODEL_PATH=/llm/models/MiniMax-H3/FL2VA \
bash scripts/minimax_h3/run_minimax_h3_t2av_tp2_ulysses4.sh
```

4 卡 TP=2 + Ulysses SP=2 + block offload 使用：

```bash
MODEL_PATH=/llm/models/MiniMax-H3/FL2VA \
bash scripts/minimax_h3/run_minimax_h3_t2av_tp2_ulysses2.sh
```

该配置面向 Intel XPU：使用 pairwise round-robin Ulysses 通信，并通过
`omni-xpu-kernel` 或 `sycl_kernels` 执行原生 XPU Flash Attention。

## CLI

在 LightX2V 仓库根目录运行：

```bash
MODEL_PATH=/data/nvme6/gushiqiao/models/MiniMax-H3 \
PROMPT='A cinematic wide shot of ocean waves at sunset, with synchronized natural ambience.' \
OUTPUT_PATH=outputs/minimax_h3_t2av.mp4 \
bash scripts/minimax_h3/run_minimax_h3_t2av.sh
```

可通过 `CONFIG_JSON`、`SEED`、`LIGHTX2V_PATH` 覆盖相应默认值。默认配置是 `configs/minimax_h3/minimax_h3_t2av.json`。

```bash
IMAGE_PATH=first.png bash scripts/minimax_h3/run_minimax_h3_i2av.sh
LAST_FRAME_PATH=last.png bash scripts/minimax_h3/run_minimax_h3_l2av.sh
IMAGE_PATH=first.png LAST_FRAME_PATH=last.png bash scripts/minimax_h3/run_minimax_h3_fl2av.sh
REFERENCES='image=person.png,video=motion.mp4,audio=voice.wav' \
  bash scripts/minimax_h3/run_minimax_h3_ref2av.sh
```

首尾帧任务在未显式传 `target_shape` 时由第一张实际输入图决定 768p 画布。`ref2av` 的参考图使用自身 2048 短边画布，参考视频重采样到 24 fps，参考音频统一为 32 kHz 立体声。

## Python pipeline

使用同一份 `config_json` 即可初始化程序化入口：

```python
import os

os.environ.setdefault("DTYPE", "BF16")

from lightx2v.pipeline import LightX2VPipeline

pipe = LightX2VPipeline(
    task="t2av",
    model_cls="minimax_h3",
    model_path="/data/nvme6/gushiqiao/models/MiniMax-H3",
)
pipe.create_generator(
    config_json="configs/minimax_h3/minimax_h3_t2av.json",
)
result = pipe.generate(
    seed=42,
    prompt="A cinematic wide shot of ocean waves at sunset, with synchronized natural ambience.",
    save_result_path="outputs/minimax_h3_t2av.mp4",
)
```

设置 `return_result_tensor=True` 时返回视频、立体声音频和采样率，不写输出文件。

Python API 的条件参数与 CLI 对应。每个 pipeline 在初始化时固定权重分区；`ref2av` 应单独初始化，并接收有序列表：

```python
ref_pipe = LightX2VPipeline(
    task="ref2av",
    model_cls="minimax_h3",
    model_path="/data/nvme6/gushiqiao/models/MiniMax-H3",
)
ref_pipe.create_generator(
    config_json="configs/minimax_h3/minimax_h3_ref2av.json",
)
ref_pipe.generate(
    prompt="Keep the character and camera motion, with matching ambience.",
    references=["image=person.png", "video=motion.mp4", "audio=voice.wav"],
    save_result_path="outputs/ref2av.mp4",
)
```

## 当前未支持

- CFG、negative prompt 条件分支和 CFG parallel
- tensor parallel、Ring sequence parallel，以及 Ulysses 与 TP/CFG parallel 的组合
- DiT 量化、LoRA 和 feature caching
- `lazy_load`、`unload_modules` 和 warmup
- 转换后的 checkpoint 或非官方 safetensors 权重格式
