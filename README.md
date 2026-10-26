# DyaDiT

训练和推理使用同一套独立的核心实现，入口为 `train.py` 和 `inference.py`，内部导入均指向本仓库的 `dyadit` 包，不依赖父目录中的训练脚本。

本版本修复了模型全局初始化覆盖两路 Wav2Vec 预训练线性层的问题，包含完整训练、断点恢复、长序列推理、SMPL-H 有声渲染、每轮动作 FID 及 GT 基准。
仓库原有文件继续保留用于对照；旧版说明和依赖分别见 [README-legacy.md](README-legacy.md) 和 [requirements-legacy.txt](requirements-legacy.txt)。日常训练和推理使用下面的新入口。

保留的流程是：双人数据 → 预训练 VQ-VAE 编码 → 双语音/动作/关系/人格条件 → DiT 扩散训练与采样 → 动作解码与 SMPL-H 导出。
新的核心实现包含 DiT 的训练和推理、所需的 VQ-VAE 编解码器，以及双人 SMPL-H 渲染；no-ORCA 消融复用同一核心入口和配置。原有 VQ-VAE 预训练等文件仍保留在仓库中。
数据和权重通过配置指定，不随代码复制。

## 文件职责

```text
dyadit/                     # 仓库根目录
├── train.py                 # 训练入口
├── inference.py             # 推理入口
├── render.py                # 双人面对面、下半身固定的 SMPL-H 视频
├── preprocess.py            # 归一化统计和可选的 .npy 数据准备
├── evaluate_gt_fid.py        # 固定参考集的 GT–GT 和 codec 重建 FID 基准
├── configs/core.yaml        # 唯一的训练/推理配置
├── qsub_4gpu.sh             # 集群 4 卡训练，包含 FID 和有声视频
├── qsub_no_orca_4gpu.sh     # 复用主训练入口，仅关闭 ORCA，独立输出和 W&B run
├── requirements.txt
├── requirements-render.txt  # 可选的渲染依赖
├── dyadit/
│   ├── model.py             # 唯一的主模型 Audio2PoseDiTDual_VQVAE
│   ├── layers.py            # 注意力、位置编码、时间/标签嵌入、DiT Block
│   ├── conditioning.py      # Wav2Vec2、动作卷积编码器、风格记忆库
│   ├── codec.py             # 时序下采样 VQ-VAE
│   ├── quantizer.py         # 残差向量量化
│   ├── data.py              # 双人配对、切片、归一化、CSV 条件
│   ├── diffusion.py         # Lightning 训练封装、扩散采样、滑窗生成
│   ├── factory.py           # 训练与推理共用的模型/codec 构造
│   ├── checkpoint.py        # 原版与抽取版权重加载
│   ├── config.py            # YAML、类型正确的覆盖参数、路径解析
│   ├── rotations.py         # 轴角和 6D 旋转转换
│   ├── export.py            # SMPL-H NPZ、音频和生成条件导出
│   ├── rendering.py         # 双人布局、下半身锁定、SMPL-H 网格生成
│   ├── render_backends.py   # 固定相机，EGL/OSMesa 与 CPU 渲染
│   ├── smplh.py             # SMPL-H 模型加载，兼容旧 Chumpy 数组
│   ├── video.py             # 视频编码和双方音频混合
│   ├── visualization.py     # 训练期间的固定验证样本生成、渲染和 W&B 视频
│   ├── fid.py               # 每轮 codec 动作 FID、固定评估协议与曲线
│   └── training.py          # 数据模块、Trainer、日志与 checkpoint
└── tests/                   # 无需联网的核心回归检查
```

## 环境和数据

```bash
cd dyadit
pip install -r requirements.txt
pip install -r requirements-render.txt  # 默认训练会定期渲染视频。
```

WAV/MP3 输入额外安装 `librosa`；W&B 已包含在训练依赖中。
模型默认从 Hugging Face 加载 `facebook/wav2vec2-base-960h`；离线环境可以将 `model.audio_model_name` 指向本地预训练模型目录。

数据目录延续原版格式：

```text
data_root/
├── motion_6d_slim/*.npy      # 优先读取；也支持 motion_6d/*.npz 或 motion/*.npz
├── audio_slim/*.npy          # 也支持 audio_npz/*.npz 或 audio/*.wav, *.mp3
├── csv/relationships.csv
├── csv/participants.csv
└── motion_6d_normalization.npz
```

动作和音频文件按 `Vxx_Sxxxx_Ixxxxxxxx_Pxxxx` 名称匹配，同一会话前缀需要两位参与者。
原始 SMPL-H 文件自动抽取上半身 13 个关节和双手 30 个关节，再转换为 258 维 6D 旋转。
缺失关系或人格标签时对应向量填零。
训练和验证共用训练集的归一化统计；验证目录由 `paths.val_data_root` 单独指定。

已有统计文件时无需重新预处理。需要准备数据时运行：

```bash
python preprocess.py --data-root /path/to/train --slim
```

该命令在指定数据目录写入统计文件，以及可选的 `motion_6d_slim`、`audio_slim` 数组。已有 slim 数组会跳过。

## 训练

所有配置内的相对路径都相对于 YAML 所在目录解析。编辑 `configs/core.yaml`，或使用可重复的 `--set`：

```bash
python train.py --config configs/core.yaml \
  --set paths.data_root=/path/to/train \
  --set paths.val_data_root=/path/to/validation \
  --set vqvae.checkpoint_path=/path/to/net_best.pth \
  --set trainer.devices=1 \
  --set data.batch_size=64
```

默认模型结构及主要训练参数沿用原主实验配置，包括 96 帧动作、4 倍时间压缩、64 维潜变量、512 维隐藏状态、4 层 Transformer、1000 个记忆条目和启用正交融合。
默认训练配置是 4 GPU、每进程 batch size 256。多卡可使用：

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/core.yaml
```

学习率调度保留原代码实际执行的按 epoch 余弦衰减。日志默认使用 W&B，project 由 `logger.project` 指定。
运行前使用 `wandb login` 或环境变量 `WANDB_API_KEY` 配置账号；无网络时可用 `WANDB_MODE=offline`，之后用 `wandb sync` 上传。
生成的有效配置、W&B 本地日志和权重分别写入 `paths.output_dir/config.yaml`、`wandb/`、`checkpoints/`。

`--resume /path/to/last.ckpt` 用于恢复本抽取版的完整训练状态。
`--weights /path/to/original.ckpt` 用于从原版或抽取版 Lightning checkpoint 初始化权重并开始新的优化过程。

### no-ORCA 消融

ORCA 对应 `model.orthogonal_fusion`：开启时，从对方音频特征中减去其沿目标音频特征的投影；关闭时跳过这一步。双向 cross-attention、音频门控融合、风格记忆库及其余条件仍保留。

从头训练的消融使用相同种子 `222`、相同数据/归一化、预训练 codec 和语音编码器、模型参数量、优化器和评估设置，仅将 `model.orthogonal_fusion=false`。
每轮固定 256 个验证样本的 FID，以及每 5 轮的 10 个 600 帧立体声视频都保持启用。
`qsub_no_orca_4gpu.sh` 调用原 `qsub_4gpu.sh`，使用独立的 `outputs/cluster/no_orca_job_<JOB_ID>` 和 W&B run `dyadit-no-orca-<JOB_ID>`。

```bash
# 普通队列，计费组 tga-iil-general，4 GPU。
qsub qsub_no_orca_4gpu.sh

# 先将 AR_ID 和 MAIN_JOB_ID 设为自己的预约和主实验编号。
qsub -ar "$AR_ID" -hold_jid "$MAIN_JOB_ID" qsub_no_orca_4gpu.sh
```

首次启动不传 `--resume` 或 `--weights`；用已训练的完整模型权重继续训练，属于切换结构后的微调，不能替代这组从头训练对照。
评估时按相同 epoch/优化步数比较两条曲线。主实验的 FID 从第 125 轮开始记录；no-ORCA 会从第 1 轮开始记录，可比曲线区间取两者共有的 epoch。
独立推理使用该消融输出目录里的有效 `config.yaml`，确保推理也关闭 ORCA。

### 每个 epoch 的动作 FID 曲线

默认每轮训练结束后，对固定的 **256 个验证样本**执行真实扩散推理，计算 `val/fid_codec`（越低越好）。
W&B 该指标的横轴定义为 `fid/epoch`；同时在 `fid/curve` 上传整条曲线图片。
本地结果位于 `paths.output_dir/metrics/fid_codec/`：`fid_codec.csv`、`fid_codec.png`、`history.json`、`protocol.json` 和真实特征统计 `reference_statistics.npz`。

这是本项目的 **codec-feature motion-FID**，不是图像 Inception FID，也不是某个 benchmark 自带评估器的 FID。
动作领域的 Fréchet 指标通常比较固定动作编码器的特征分布，参见 [FGD 官方实现](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context)。这里具体采用：

- 复制并冻结预训练 VQ-VAE 的量化前 encoder，真实动作和解码后的生成动作都通过同一个 float32 encoder。
- 每个片段的时间维均值、标准差拼接为 128 维特征，不使用正在训练的 DiT 特征，也不直接比较采样 latent 和真实 encoder 特征。
- 默认评估 96 帧片段，采样 50 步、batch size 32；固定样本索引、风格片段和采样噪声。起始参考的 16 帧从真实/生成两侧同时排除。
- 使用 float64 均值/协方差和对称 PSD 矩阵平方根，支持秩不足的协方差；实际样本不足配置数量时使用全部样本，但至少需要 2 个。

该值用于同一特征编码器、数据和采样协议下的训练趋势，**不能直接与使用其他评估器的论文数值比较**；有限样本 FID 也有采样偏差。
验证输入和真实特征在启动时缓存一次，避免每轮重新解压音频；每轮 FID 仍会增加生成耗时。
600 帧、每 5 轮的 10 个视频可视化保持独立，不用这 10 个视频估计 FID。

```yaml
fid:
  enabled: true
  every_n_epochs: 1
  num_samples: 256
  batch_size: 32
  pose_length: 96
  inference_steps: 50
```

曲线会在下一次完整 epoch 的评估后出现，不伪造或插值补齐过去未评估的 epoch。
断点恢复保留已评估 epoch 和曲线历史；协议中保存编码器/输入哈希、样本、长度、前缀及采样设置。
为了保证纵向比较，修改这些设置后应使用新的输出目录/实验，不能直接拼在已有曲线上。
FID 不改变优化目标或 checkpoint 的 `val/loss` 选择规则，也不会消耗后续训练的随机数状态。

GT 基准可以独立在 CPU 上计算，不需要加载 DiT、不需要重启训练：

```bash
python evaluate_gt_fid.py \
  --run-dir /path/to/full-model-run \
  --compare-run-dir /path/to/no-orca-run
```

保留训练评估中固定的 256 个真实片段作为参照，从其余验证片段抽取另一组 256 个，重复 100 次，保存 GT–GT FID 的均值、标准差和 2.5%–97.5% 分位范围。
两组片段索引互不重叠，但可来自相同会话；该范围描述固定参照下的抽样波动，不是置信区间或生成模型的严格下界。
同一组 GT 与自身比较的 FID 应接近 0，只作为数值自检。额外计算完整 VQ 编解码重建后的 FID，作为 codec 重建误差的参照。
编码器哈希、片段顺序及缓存真实特征均与训练协议核对；特征提取、排除前 16 帧和样本数保持一致。
结果默认保存在 `run-dir/metrics/fid_codec/gt_baseline/`，包括 `result.json`、100 次抽样 CSV、特征 NPZ 和带基准线的实验曲线 PNG。
加 `--wandb` 上传到独立的基准实验；已完成本地计算时可用 `--wandb --publish-only` 上传结果。该实验保存上传时两条训练曲线的快照，不复用训练 run 的日志步数。

### 每 5 个 epoch 的 W&B 视频

默认在第 **5、10、15…个完整 epoch 结束后**，从验证集固定选择 **10 个不同样本**，用当前权重执行扩散采样，再上传带双方原声的 MP4 到 W&B 的 `visualization/samples`。
视频中左侧蓝色人物是对方 B 的真实条件动作，右侧橙色人物是模型生成的 A；两人面对面、下半身固定，沿用推理时的目标动作前缀条件。
声音按人物分成立体声：蓝色人物的原声只放左声道，橙色人物的原声只放右声道。
默认使用与独立推理相同的采样器，生成 **600 帧（20 秒）**的长序列：窗口 96 帧、步长 80 帧，前后重叠 16 帧。
长序列从单独的验证数据切片中读取，不拼接、不循环播放原来的 96 帧片段；训练和验证损失仍使用 96 帧，不改变训练 batch。
视频为 640×480、15 fps（600 帧动作采样为 300 帧视频），双方音频仍是完整同步的 20 秒，并统一调整响度。
W&B 视频播放器可能默认静音，点击声音按钮即可收听。

配置位于 `visualization`，当前模型路径使用本机已验证的 SMPL-H 目录。迁移环境时修改该路径：

```bash
python train.py --config configs/core.yaml \
  --set visualization.model_path=/path/to/smplh_models \
  --set visualization.every_n_epochs=5 \
  --set visualization.num_samples=10 \
  --set visualization.pose_length=600 \
  --set visualization.window_frames=96 \
  --set visualization.stride_frames=80
```

视频及对应的动作 NPZ、双方 WAV、条件元数据保存在：

```text
paths.output_dir/visualizations/
├── epoch_0005/
│   ├── manifest.json
│   └── sample_XXXXX/conversation.mp4  # 同目录保留动作、原声及首帧预览。
└── epoch_0010/...
```

固定长序列验证索引和采样噪声便于跨 epoch 比较，不改变训练的随机数状态；若覆盖 600 帧的验证样本不足 10 个，则全部渲染，不重复凑数。
`manifest.json` 与 W&B 视频标题记录序列长度和滑窗参数。没有覆盖目标长度的验证对话时启动即报错，不会用短片段补齐。
多卡仅 global rank 0 生成、渲染和上传，其余进程等待；默认 SMPL-H 网格和渲染使用 CPU，不占用训练 GPU 的网格内存。
生成使用训练的混合精度设置；渲染结束后恢复模型的训练/冻结状态。600 帧长序列会增加每轮预览的采样与渲染耗时。
sanity check、`fast_dev_run` 不渲染；恢复 checkpoint 会保留已输出的 epoch 状态。
缺少 W&B logger、渲染依赖或 SMPL-H 路径时启动即报错，避免训练到第 5 轮才发现配置遗漏。

需要纯训练或 TensorBoard 时显式关闭视频：

```bash
python train.py --set visualization.enabled=false --set fid.enabled=false --set logger.type=tensorboard
```

### TSUBAME 4 卡排队

`qsub_4gpu.sh` 使用 `tga-iil-general` 组申请一个完整的 4 GPU 节点，时限 24 小时（该组单任务上限）。
脚本使用本机已验证的数据、VQ-VAE 权重和离线 Wav2Vec2 缓存，不复制或硬编码 W&B 密钥。
必须从当前仓库根目录提交，脚本通过 `SGE_O_WORKDIR` 运行本次 checkout 的代码。可用 `DYADIT_PYTHON`、`DYADIT_TRAIN_ROOT`、`DYADIT_VAL_ROOT`、`DYADIT_CODEC`、`DYADIT_AUDIO_MODEL` 覆盖机器路径；用 `WANDB_ENTITY`、`WANDB_PROJECT` 覆盖 W&B 目标，通过 `qsub -v` 传入。

```bash
cd dyadit
mkdir -p outputs/qsub_logs
qsub qsub_4gpu.sh
```

任务日志位于 `outputs/qsub_logs/dyadit_core_4g.o<JOB_ID>`，训练输出位于 `outputs/cluster/job_<JOB_ID>/`。
W&B 上传到 `ohicarip/wan_dit_audio2pose`，run ID 为 `dyadit-core-<JOB_ID>`，每 5 轮保存 10 个带声音的视频。
如果到达时限，最近一次完整 epoch 的权重仍保存在该任务目录的 `checkpoints/last.ckpt`；恢复时可把 `--resume /path/to/last.ckpt` 作为提交脚本的参数。
需要继续同一个输出目录和 W&B 实验时，可在提交环境中设置并用 `qsub -v DYADIT_OUTPUT_DIR,DYADIT_RUN_NAME,WANDB_RUN_ID ...` 传入这些变量；不传时仍为每个任务创建独立目录和实验。
正在运行的进程不会自动重载代码或 YAML；修改可视化长度后，需要从 checkpoint 恢复才能生效。已上传的旧短视频会保留。

## 推理

```bash
python inference.py --config configs/core.yaml \
  --checkpoint /path/to/dyadit.ckpt \
  --set paths.data_root=/path/to/train \
  --set vqvae.checkpoint_path=/path/to/net_best.pth \
  --data-root /path/to/test \
  --index 0 --num-samples 1 \
  --pose-length 240 --window-frames 96 --stride-frames 80 \
  --inference-steps 50 --output-dir inference_outputs
```

`--data-root` 选择推理数据，归一化仍使用配置指定的训练统计。
`--pose-length`、`--window-frames`、`--stride-frames` 均以原始动作帧为单位，且需要能被 VQ-VAE 的时间压缩倍数整除。
`model.mask_length` 以潜变量步数为单位；默认 4 步对应 16 帧动作。

默认沿用原版的数据集条件生成方式：双方语音、对方动作、同人物风格动作、关系、人格都作为条件，并使用目标动作的起始前缀。
可用 `--no-reference-prefix` 关闭目标动作前缀，但仍会使用对方动作条件。
`--relationship 1 0 0 0` 和 `--personality 1 0.8 0.6 0.4 0.2` 可显式设置条件；人格在训练中的取值为原分数除以 5。

每个样本导出生成动作、原始目标动作、对方动作的 SMPL-H NPZ、双方 WAV、生成潜变量以及 `metadata.json`。
下半身、全局旋转和位移沿用原导出约定，填默认值。

## 双人面对面渲染

```bash
pip install -r requirements-render.txt

python render.py \
  --sample-dir inference_outputs/sample_00000 \
  --model-path /path/to/smplh_models
```

默认人物与声音绑定如下（导出文件中的模型 A/B 身份不变，只调整显示位置）：

| 显示人物 | 动作来源 | 对应原声 | 声道 |
| --- | --- | --- | --- |
| 左侧蓝色：条件人物 B | `target_motion_other_denorm.npz` | `input_audio_other.wav` | 左声道 |
| 右侧橙色：生成人物 A | `generated_motion_denorm.npz` | `input_audio.wav` | 右声道 |

两人位于同一场景、相向站立，骨盆间距默认 1.6 米。相机固定，并显示地面。
默认输出 AAC 立体声，不会把双方声音混成单声道，也不把同一混音复制到左右两侧。某一方音频缺失时，其对应声道保持静音，不交换声道。
默认对双声道统一做响度归一化，避免原录音过轻；保留两人的相对音量，不单独放大监听方的背景噪声，也不会混合声道。
需要原始音量时可加 `--no-audio-normalize`；需要原来的不分人物混音时可加 `--audio-layout mix`。
生成 `conversation.mp4`、首帧预览 `conversation.png` 和记录相机/帧率等设置的 `conversation.json`。

也可以指定两个人各自的动作文件，包括两份生成结果：

```bash
python render.py \
  --motion-a /path/to/person_a.npz --motion-b /path/to/person_b.npz \
  --audio-a /path/to/person_a.wav --audio-b /path/to/person_b.wav \
  --model-path /path/to/smplh_models --output outputs/conversation.mp4
```

模型路径支持单个 `.pkl` / `.npz` 文件，以及 `smplh/SMPLH_MALE.pkl`、`SMPLH_male.pkl`、`male/model.npz` 等目录布局。
当前机器可使用 `/gs/bs/tga-koike-crescent/EasyMocap/data/smplx/smplh`，已验证其中的男版模型。
显式文件渲染时，`--motion-a/--audio-a/--gender-a` 指左侧蓝色人物，`--motion-b/--audio-b/--gender-b` 指右侧橙色人物。
性别通过 `--gender-a male --gender-b female` 分别选择；默认均为男版模型，且使用各自动作文件中的静态体型参数（没有时为零）。
旧 Chumpy 文件中的普通数组和 Select 切片会直接转成数值数据读取，不需要安装 Chumpy，也不会改写模型文件。

下半身固定包括：

- 默认 `--lower-body neutral` 将髋、膝、踝、脚部关节保持为中立姿态；`--lower-body first-frame` 则保持各自第一帧的下肢姿态。
- 忽略输入中随时间变化的全局旋转和位移，使用固定的两人位置和相向旋转。
- 固定骨盆/腿脚占主导蒙皮权重的顶点，避免 SMPL-H 姿态修正项造成腿脚轻微漂移；足底高度固定在地面。
- 上半身、头部和双手仍使用输入动作。SMPL-H 在这里没有独立的口型/表情输入；带音频的说话手势可以直接渲染，口型需要额外的面部模型和参数。

`--backend auto` 优先尝试 EGL 或 OSMesa，不能创建 OpenGL 上下文时自动转为纯 CPU 三角形渲染。
CPU 节点可以直接指定 `--backend software`。GPU 网格推理可以另设 `--device cuda`。
`--separation` 设置两人间距，`--width/--height` 设置分辨率，`--max-frames 96` 限制预览长度。
`--step 2` 将 30 fps 动作抽为 15 fps 视频，保持原播放时长与音频同步；`--start-frame` 同步裁剪动作和音频。
`--save-frames` 可额外导出 PNG。已有输出默认不覆盖，需要时显式使用 `--overwrite`。
手部默认 `--hand-mean flat`，直接解释导出的轴角；只有输入是相对于 MANO 均值的残差时才用 `--hand-mean model`。

推理后自动渲染可在原推理命令后追加：

```bash
--render --smplh-model /path/to/smplh_models --render-backend auto
```

## 兼容性与清理中修正的行为

主模型层级和参数名称保留，能够严格加载原主模型的 `model.*`、`vqvae.*` 权重。
旧 checkpoint 元数据中的训练配置和 codec 类名由本目录兼容加载，不需要导入父目录代码。
模型的注意力计算统一使用 PyTorch SDPA，去掉可选 FlashAttention 的导入分支。

抽取版同时修正了影响主流程的连接和单位问题：

- 训练、推理共用 `factory.py`，`orthogonal_fusion` 在两边都读取同一配置。
- 模型初始化跳过两路已加载的 Wav2Vec 子模块，保留预训练权重。2026-10-26 排查发现，旧版全局线性层初始化会覆盖它们的 14 个线性层；该问题来自原模型并在抽取时保留，现已修复。旧 checkpoint 严格加载仍使用其已训练权重；从旧 checkpoint 续训不会自动恢复预训练初始化。检验修复后的训练效果需要另起实验。
- 对方动作条件保持 `[B, T, C]`，没有整理包中额外的维度扩展。
- 推理保留传入的风格动作条件。
- 滑窗按原始动作帧切音频，在潜变量空间按对应压缩倍率拼接，最后统一解码。
- 采样器支持 DDPM、DDIM、DPM-Solver；CFG 在三种调度器上都执行。默认 `guidance_scale=1`，对应原 DPM-Solver 实际未使用 CFG 的行为。
- CFG 的无条件分支清空交叉注意力 context，与训练时的 `fusion_drop` 一致；时间/关系/人格的调制保持不变。
- 训练和验证使用一致的 mask 输入与全序列噪声损失。旧版验证漏传 mask 并排除前缀，其历史验证数值不应直接比较。
- `samples_per_conversation=-1` 按每个会话实际长度建立不重叠切片，`is_reversed` 开关生效；这修正了原版不同长度会话共用平均切片数量的问题。

因此可以复用原参数，但这些连接与采样修正意味着生成结果不承诺逐位复现旧推理脚本。

## 检查

```bash
pip install pytest
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

测试使用可控的小型语音编码器，覆盖实际核心模型前向/反向、正交融合、codec、三种采样调度器、滑窗时间对齐、旧 checkpoint 元数据兼容、结果导出，以及周期性预览的触发/恢复和带音频的 W&B 离线视频，不会下载模型或向 W&B 上传测试数据。
