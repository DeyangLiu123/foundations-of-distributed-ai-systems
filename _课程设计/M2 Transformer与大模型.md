---
title: 设计稿 · M2 Transformer与大模型
tags:
  - course/design
---

# M2 Transformer 与大模型 · 设计稿

> 模块目标：把 LLM 从黑盒拆成**可以计账的白盒**。学完读者应能：看懂任意模型的 config.json、手算参数量与 KV cache、看懂技术报告里的训练/后训练流程描述。系统视角贯穿：每讲一个结构，都问「它的计算/显存/通信形状是什么」。

## L10 Token与嵌入

- **文件**：`M2 Transformer与大模型/L10 Token与嵌入.md`
- **定位**：LLM 数据表示的地基。讲清「token 是本领域的通用货币」——计价、计吞吐、计上下文全用它。
- **前置**：[[L04 神经网络与前向传播]]
- **核心问题**：① 文字怎么变成神经网络能算的数字？② 为什么是 token 而不是单词/字母？③ embedding 是什么，占多少参数？
- **内容要点**：
  1. **tokenization**：字符级/词级的缺陷 → 子词方案 **BPE**（合并高频对的直觉演示，用「unhappiness → un+happiness」类例子）；**vocabulary** 与 vocab size 的量级（32K–256K）。
  2. 中英文差异：1 英文词 ≈ 1.3 token、1 汉字 ≈ 0.5–1 token（tokenizer 相关）；「按 token 计费」的 API 经济学。
  3. **embedding**：token id → 查表得 $d$ 维向量；embedding matrix 形状 $V \times d$；「语义即几何」一段直觉（相似词向量相近）——顺带预告 RAG 里的 embedding model（L68）。
  4. **context window**：模型一次能处理的 token 上限；它不是硬件常数而是训练决定的（细节 L18）；8K/128K/1M 的量级感。
  5. 特殊 token：BOS/EOS/PAD、chat template 一句话（system/user/assistant 标记）。
  6. 系统接口：tokens/s 作为训练与推理吞吐的通用单位；tokenizer 在数据管线中的位置（预处理离线做，L34）。
- **必收术语**：token、tokenize/tokenizer、BPE、subword、vocabulary/vocab size、embedding、embedding matrix、one-hot（一句话）、context window/context length、special token、BOS/EOS、chat template、tokens per second。
- **定量环节**：Llama-3 embedding 参数：128256 × 4096 ≈ 5.25×10⁸（0.53B）——占 8B 模型的 6.5%；再算一本 50 万字中文小说 ≈ 30–50 万 token，与 128K context window 对比。
- **图示**：①「你好世界 → token ids → 向量序列」流水线图；② embedding 查表示意。
- **延伸阅读**：OpenAI tokenizer playground / tiktoken（动手感受分词）；Karpathy《Let's build the GPT Tokenizer》视频。
- **误区**：「token = 单词」；「context window 是显存决定的」——显存决定的是 KV cache 上限，窗口本身是模型属性。

## L11 注意力机制

- **文件**：`M2 Transformer与大模型/L11 注意力机制.md`
- **定位**：全课程最重要的单个机制。要求讲到「读者能默写 attention 公式并说出每个矩阵的形状」，同时给出 $O(S^2)$ 复杂度的系统后果。
- **前置**：[[L10 Token与嵌入]]
- **核心问题**：① attention 到底在「注意」什么？② Q、K、V 三个矩阵从哪来、形状是什么？③ 为什么注意力是平方复杂度，后果是什么？
- **内容要点**：
  1. 动机：一个词的含义取决于上下文（bank 河岸/银行）→ 需要「每个位置查看其他所有位置」的机制。
  2. 检索类比：**Query**（我要找什么）、**Key**（我是什么标签）、**Value**（我的内容）；数据库/图书馆类比展开。
  3. 公式逐步构建：$QK^T$（相似度打分）→ 除 $\sqrt{d_{head}}$（防梯度问题，一句话）→ softmax（分配注意力权重）→ 乘 $V$（加权取内容）。**全程标形状**：$[S,d_h]\times[d_h,S]\to[S,S]$ → $[S,S]\times[S,d_h]\to[S,d_h]$。
  4. **multi-head**：多组 QKV 并行、各看各的关系（语法头/指代头直觉）；头数 $h$、$d_{head}=d/h$；拼接后过输出投影 $W_O$。
  5. **causal mask**：语言模型只许看左边（下三角掩码）；self-attention vs cross-attention 一句话。
  6. 系统后果：注意力矩阵 $S\times S$ → 计算与（朴素实现的）显存都是 $O(S^2)$；S=128K 时 attention matrix 有 160 亿个元素 → 这是 FlashAttention（L51）、长上下文并行（L45）、线性注意力（L18）三条研究线的共同起点。
- **必收术语**：attention、self-attention、cross-attention、query/key/value（QKV）、attention score/weight、scaled dot-product attention、multi-head attention（MHA）、attention head、$d_{head}$、causal mask、bidirectional vs causal、quadratic complexity。
- **定量环节**：单层 attention 的 FLOPs 与显存：S=8192、d=4096、h=32：QKV 投影 GEMM 的 FLOPs vs $QK^T$+$AV$ 的 FLOPs（$4S^2d$ 量级）比值 = $S/d$ 关系 → 得出「短序列时投影主导、长序列时二次项主导」的分水岭 S≈d。朴素 attention matrix 显存：$h \times S^2$ × 2 B（BF16）= 32×8192²×2 ≈ 4.3 GB/层 → 不可行，引出 FlashAttention 的必要性。
- **图示**：① QKV 检索类比图；② scaled dot-product 数据流图（标形状，本课主图）；③ causal mask 下三角示意。
- **延伸阅读**：Jay Alammar《The Illustrated Transformer》（有中译）；3Blue1Brown《Attention in transformers, visually explained》；bbycroft.net/llm（3D 交互可视化，强烈推荐）。
- **误区**：「Q=K=V」——它们是同一输入乘三个不同投影矩阵的结果；「多头 = 多次重复计算」——是把 $d$ 切成 $h$ 份并行，总计算量基本不变。

## L12 Transformer全解剖

- **文件**：`M2 Transformer与大模型/L12 Transformer全解剖.md`
- **定位**：本模块的「工程图纸」课。目标只有一个动作：**给读者一份 Llama-3-8B 的 config.json，逐字段讲懂，然后手算出 8.03B 参数**。做完这件事，任何模型卡都能读。
- **前置**：[[L11 注意力机制]]
- **核心问题**：① 一层 Transformer 里除了 attention 还有什么？② 参数都藏在哪（答案：大头在 FFN）？③ config.json 每个字段什么意思？
- **内容要点**：
  1. 一层 decoder block 的完整数据流：input → **RMSNorm** → attention → **residual add** → RMSNorm → **FFN** → residual add。**pre-norm** vs post-norm 一句话（现代模型全是 pre-norm，训练更稳）。
  2. **FFN/MLP block**：up-projection → 激活（**SwiGLU**：gate+up 两个投影）→ down-projection；$d_{ff} \approx 3.5d$（SwiGLU 惯例）；**参数大头在这里**（约 2/3）。
  3. **RMSNorm**（对比 LayerNorm 一句话：去掉均值中心化，更省）；residual connection 回收 L08——梯度高速公路。
  4. 位置信息：为什么 attention 天生不知道顺序 → 绝对位置编码（一句话）→ **RoPE**（旋转位置编码，给「旋转角度编码相对位置」直觉即可，数学不展开；长上下文外推预告 L18）。
  5. 输出端：最后的 RMSNorm → **lm_head**（$d \times V$ 投影回词表）；**weight tying** 一句话。
  6. 主任务：Llama-3-8B config.json 逐字段（hidden_size=4096, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336, vocab_size=128256, …）→ 手算参数量（见定量环节）。GQA 字段此处只说「KV 头更少，L18 细讲」。
  7. dense vs MoE 预告（L17）；encoder-only（BERT）/decoder-only（GPT）/encoder-decoder（T5）三家族一段话收尾。
- **必收术语**：Transformer block/layer、decoder-only、FFN/MLP block、up/down/gate projection、SwiGLU、LayerNorm、RMSNorm、pre-norm、residual stream、positional encoding、RoPE、lm_head、weight tying、config.json、hidden size、intermediate size、num_layers、model card、dense model。
- **定量环节**（本课灵魂，务必完整呈现）：Llama-3-8B 参数手算：embedding 128256×4096≈0.525B；每层 attention（GQA）：$W_Q$ 4096²+$W_K$,$W_V$ 各 4096×1024+$W_O$ 4096² ≈ 41.9M；每层 FFN：3×4096×14336 ≈ 176.2M；单层合计 ≈ 218M ×32 层 ≈ 6.98B；+ lm_head 0.525B + norm 零头 → **≈ 8.03B** ✅。附「参数分布饼图」：FFN 65% / attention 15% / embedding+lm_head 13%。
- **图示**：① decoder block 数据流图（本课程被引用最多的一张图，画精细）；② 参数分布饼图/条形图。
- **延伸阅读**：Llama 3 技术报告《The Llama 3 Herd of Models》第 3 节（结构表）；HuggingFace 上任一模型的 config.json（布置作业：算 Qwen 某个尺寸）。
- **误区**：「参数大头在 attention」——在 FFN；「7B/8B 是精确值」——是四舍五入的营销数，手算会差零头；「层数越深越强」——宽深比是权衡（一句话即可）。

## L13 自回归生成与KV缓存

- **文件**：`M2 Transformer与大模型/L13 自回归生成与KV缓存.md`
- **定位**：**通往整个 M8 推理模块的桥**。prefill/decode 两阶段 + KV cache 推导，是推理论文 100% 会出现的背景知识，必须在这里一次讲透。
- **前置**：[[L12 Transformer全解剖]]
- **核心问题**：① 模型怎么「一个字一个字往外蹦」？② KV cache 从哪推导出来的、有多大？③ 为什么说 prefill 和 decode 是两种完全不同的计算？
- **内容要点**：
  1. **autoregressive generation**：下一个 token 的概率分布 → 采样 → 拼回输入 → 再来一轮；停止条件（EOS/max length）。
  2. **sampling 家族**：greedy、**temperature**（放大/压平分布）、top-k、**top-p**（nucleus）；beam search 一句话（LLM 时代少用）；「为什么同一问题两次回答不同」。
  3. 朴素生成的浪费：每生成一个 token 都全量重算历史 → $O(S^2)$ 每 token；**观察：causal mask 下历史 token 的 K/V 永远不变** → 缓存之 → **KV cache**。逐步推导，这是本课高光。
  4. **prefill vs decode**：prefill = 一次并行算完 prompt（大 GEMM，compute-bound）；decode = 每步只算 1 个新 token 但要读全部 KV cache 和权重（memory-bound）。给两阶段的直觉对比表（计算形状/瓶颈/对应指标 TTFT/TPOT——指标名此处首次提及，深入 L55）。
  5. KV cache 大小公式：$2 \times L \times S \times h_{kv} \times d_{head} \times bytes \times batch$（进 [[03 约定与符号]] 口径）；随 batch 和 S 线性爆炸 → 预告 PagedAttention（L56）与 GQA/MLA（L18）。
  6. **perplexity** 一段话（loss 的指数，语言模型质量的传统度量）。
- **必收术语**：autoregressive、next-token prediction、decoding、sampling、greedy decoding、temperature、top-k/top-p（nucleus）sampling、EOS、max new tokens、KV cache、prefill、decode phase、TTFT/TPOT（预告级）、perplexity、streaming（逐 token 返回）。
- **定量环节**：Llama-3-70B、BF16、S=8192、batch=1 的 KV cache：2×80×8192×8×128×2 B ≈ **2.7 GB**；若是 MHA（64 头）则 21.5 GB——GQA 省 8×。再算 batch=32 时 86 GB > H100 显存 → 引出「KV cache 才是推理并发瓶颈」。
- **图示**：① 自回归循环图（生成→拼接→再生成）；② prefill/decode 两阶段时序图（大方块 + 一串小方块，此图 M8 反复用）；③ KV cache 逐步长大的示意。
- **延伸阅读**：《Transformer Inference Arithmetic》（kipply 博客，推理计账经典）；HuggingFace 博客关于 KV cache 的图解文章。
- **误区**：「decode 慢是因为算得多」——恰恰是算得少、搬得多（memory-bound，L23/L55 定量证明）；「KV cache 是可选优化」——没有它每 token 成本随长度平方增长，是必需品。

## L14 预训练

- **文件**：`M2 Transformer与大模型/L14 预训练.md`
- **定位**：训练侧的全流程叙事课。让读者能看懂技术报告的 pretraining 章节和训练日志截图，为 M6/M7 的「为什么要这些系统」提供 workload 直觉。
- **前置**：[[L13 自回归生成与KV缓存]]、[[L07 训练循环解剖]]
- **核心问题**：① 15T token 从哪来、怎么洗？② 预训练的 loss 是什么、曲线长什么样？③ 技术报告里的 benchmark 表怎么看？
- **内容要点**：
  1. 预训练 = 互联网规模的 next-token prediction；self-supervised 的含义（数据自带标签）。
  2. 数据管线：**CommonCrawl** 等来源 → 抽取/过滤（质量分类器）→ **deduplication** → 混配（**data mixture**：代码/多语言/数学比例）→ tokenize 成 shards；数据决定模型上限的一段议论；**data curriculum/annealing**（收尾阶段喂高质量数据）一句话。
  3. 训练日程：loss 曲线的形状（快降→缓降幂律）；**loss spike** 与处置（跳过数据/回滚 ckpt，深入 L49/L53）；lr schedule 回收 L06；训练中评测（**held-out loss** 与下游 benchmark）。
  4. **benchmark 扫盲**：MMLU、GSM8K、HumanEval 各一句话（读者只需认识名字）；「跑分 ≠ 好用」的一段提醒；**contamination** 一句话。
  5. workload 画像（给后续模块的接口）：数万卡 × 数月、纯吞吐导向、全同步步调（每 step 全局同步一次梯度——这就是 M5/M6 的主角场景）、失败重来代价巨大（M7 的主角场景）。
  6. 案例串讲：以 Llama-3 405B 为例过一遍（15.6T token、~16K H100、54 天量级、GBS/上下文分阶段——数字以报告为准，标注「见原文」）。
- **必收术语**：pretraining、corpus、CommonCrawl、data cleaning/filtering、deduplication、data mixture、data curriculum、annealing、shard、held-out/validation loss、benchmark、MMLU（认名即可）、contamination、base model、training run、loss spike、token budget。
- **定量环节**：数据存储账：15T token ≈ 30 TB（2B/token 存 token id）+ 原始网页数 PB 量级；再算一天吞吐：16K 卡 × 400 TFLOPS 有效 ÷ (6×405B) ≈ 2.3×10⁹ token/天？——期望实现者按 6ND 完整算出「天数×卡数」与报告对齐（结果 ~50–70 天量级即算对）。
- **图示**：① 数据管线漏斗图（原始 PB → 清洗后 XX TB → 15T token）；② 典型 loss 曲线（标注 spike、annealing 阶段）。
- **延伸阅读**：《The Llama 3 Herd of Models》第 2–3 节；FineWeb 数据集博客（HuggingFace，数据清洗的现代标准流程）。
- **误区**：「数据越多越好」——去重和质量过滤常常删掉 90%+；「loss 越低模型越好」——不同数据分布的 loss 不可比。

## L15 后训练与对齐

- **文件**：`M2 Transformer与大模型/L15 后训练与对齐.md`
- **定位**：从 base model 到「会聊天的助手」。术语密度极高的一课（SFT/RLHF/DPO/GRPO/LoRA/CoT…），目标是**认识流程与名词**，系统实现留给 L54。
- **前置**：[[L14 预训练]]
- **核心问题**：① base model 为什么不能直接用？② RLHF 到底在优化什么？③ 2025 年后爆火的 reasoning model 是怎么训出来的？
- **内容要点**：
  1. base model 的行为（只会续写）→ 需要**对齐（alignment）**到人类意图。后训练三段式总图：**SFT → 偏好优化（RLHF/DPO）→（可选）RL for reasoning**。
  2. **SFT**（instruction tuning）：高质量问答对上继续训练；数据从人写到蒸馏（**distillation** 概念顺带正式定义）。
  3. **RLHF**：reward model（从人类偏好对学打分）→ **PPO** 优化 policy（每个词一句话级别：actor/critic/reference model/KL 惩罚——只建立名词地图，流程图给全）；**DPO**（跳过 RM 的简化版，一句话原理）。
  4. **GRPO 与 reasoning model**：可验证奖励（数学/代码）+ 大规模 RL → **long CoT**「思考」行为涌现；o1/DeepSeek-R1 时刻；**test-time compute/scaling** 概念（推理时多想 = 变相加算力）——顺带指出它对 serving 的冲击（输出变长 10×，L55/L61 回收）。
  5. **PEFT/LoRA**：低秩旁路省显存的直觉 + 「只存 adapter」对 serving 的意义（multi-LoRA，L61 回收）。
  6. 系统接口预告：RLHF = 训练引擎 + 推理引擎在同一作业里共存（rollout 生成用推理、更新用训练）→ 这是 L54 的主角问题。
- **必收术语**：post-training、alignment、SFT/instruction tuning、preference data、reward model、RLHF、PPO（认名）、reference model、KL penalty、DPO、GRPO、RLVR/verifiable reward、reasoning model、chain-of-thought（CoT）、long CoT、test-time compute、distillation、PEFT、LoRA、adapter、catastrophic forgetting（一句话）。
- **定量环节**：LoRA 参数账：8B 模型全量微调需 16B/参数 ≈ 128 GB 训练状态；LoRA rank=16 只训 ~0.2% 参数 → 优化器状态缩到 GB 以下，单卡可跑。算出具体数字。
- **图示**：① 后训练三段式流程图（base→SFT→RLHF→reasoning RL）；② RLHF 四模型关系图（actor/critic/RM/ref，此图 L54 直接复用）。
- **延伸阅读**：InstructGPT 论文《Training language models to follow instructions with human feedback》（NeurIPS 2022，看图 2）；DeepSeek-R1 技术报告（体会 RLVR 叙事）；HuggingFace RLHF 图解博客。
- **误区**：「RLHF 让模型更聪明」——主要是更「听话」，能力大头来自预训练；「reasoning model 只是 prompt 技巧」——是 RL 训练出的行为，且真金白银消耗推理算力。

## L16 Scaling Law与算力账

- **文件**：`M2 Transformer与大模型/L16 Scaling Law与算力账.md`
- **定位**：全课程定量骨架课。**6ND 公式在此正式推导**，Chinchilla 定律在此讲清。学完读者能对任何训练报告做「算力审计」。
- **前置**：[[L12 Transformer全解剖]]、[[L05 反向传播与梯度]]
- **核心问题**：① 为什么训练 FLOPs ≈ 6ND？② 给定算力预算，模型该多大、数据该多少？③ 「X 万卡训 Y 天」这类新闻怎么核实？
- **内容要点**：
  1. **6ND 推导**（本课高光）：前向每参数每 token 2 FLOPs（一次乘加）→ 2N；反向 ≈ 2× 前向 → 4N；合计 6N/token，×D 个 token。注意力 $O(S^2)$ 修正项：给出「S 不太长时可忽略，长序列需加回」的判断标准（如 S 与 6d 的比较，具体系数**须核实**后写，来源 PaLM 附录 B 或 Chinchilla 附录——实现者查证后引用，查不到就定性）。
  2. **scaling law**：loss 随 N、D、C 的幂律下降（Kaplan 2020）；**Chinchilla**（2022）修正：算力最优配比 **D ≈ 20N**；「过训练」趋势（Llama 系 D/N 达 200+：训练多花算力换推理省算力——**训练-推理算力权衡**，系统含义大）。
  3. 算力单位阶梯：FLOPs → GPU-hours → 台数×天数 → 美元 → 兆瓦。给换算链演示。
  4. **emergent abilities** 一段话（含「可能是度量假象」的争议，保持克制）。
  5. 实战：核算三个公开案例（Llama-3-405B ≈ 3.8×10²⁵ FLOPs；GPT-3 175B×300B token ≈ 3.15×10²³；DeepSeek-V3 的 2.788M H800-hours——注意 MoE 用**激活参数** 37B 计算），教「审计新闻数字」的姿势。
- **必收术语**：scaling law、power law、compute budget（C ≈ 6ND）、compute-optimal、Chinchilla、over-training、tokens-per-parameter（D/N）、GPU-hours、emergent abilities、training-inference tradeoff、effective FLOPS、（复习）MFU。
- **定量环节**：完整算一遍 Llama-3-405B：6×4.05×10¹¹×1.56×10¹³ ≈ 3.8×10²⁵ FLOPs；÷（16384 卡 × 989 TFLOPS × 40% MFU）≈ 5.9×10⁶ s ≈ **68 天**——与公开信息同数量级 ✅。再让读者反算：同预算下 Chinchilla 最优的 N 和 D 是多少。
- **图示**：① loss-compute 幂律示意（对数轴）；② 「N×D 平面上的等算力线 + Chinchilla 最优线 + 过训练区」示意图。
- **延伸阅读**：《Training Compute-Optimal Large Language Models》（Chinchilla，NeurIPS 2022，看图 1/图 3）；《Scaling Laws for Neural Language Models》（Kaplan et al. 2020，选读）；Epoch AI 的算力数据库（数字核查工具）。
- **误区**：「参数越多越强」——数据不够时是浪费（Chinchilla 的核心论点）；MoE 模型用总参数算 FLOPs（应该用激活参数）；把稀疏峰值算力代入 MFU 分母。

## L17 MoE混合专家

- **文件**：`M2 Transformer与大模型/L17 MoE混合专家.md`
- **定位**：DeepSeek 时代读论文的必备结构知识。本课讲**模型结构与直觉**，训练系统（EP/all-to-all 工程）留给 L46，推理留给 L60。
- **前置**：[[L12 Transformer全解剖]]、[[L16 Scaling Law与算力账]]
- **核心问题**：① 「6710 亿参数但只激活 370 亿」是什么意思？② router 怎么决定 token 去哪个专家？③ 为什么 MoE 对网络特别不友好？
- **内容要点**：
  1. 动机：scaling law 说参数越多越好，但 FLOPs 跟着涨 → **稀疏激活**：参数多而每 token 只用一小撮 → **MoE**：把 FFN 复制成 E 份「专家」，每 token 挑 top-k 份。
  2. 结构：**router/gate**（一个小线性层打分）→ **top-k 选择** → 加权组合专家输出；**shared expert**（人人都过，DeepSeek 风格）；只替换 FFN、attention 保持稠密（主流做法）。
  3. 麻烦一：**负载不均**——热门专家挤爆 → **auxiliary loss**（负载均衡辅助损失）/ **aux-free bias**（DeepSeek-V3 的偏置调节法，一句话）；**capacity factor** 与 **token dropping**（vs **dropless**）。
  4. 麻烦二（系统预告）：专家分布在不同 GPU 上（**expert parallelism**）→ 每层两次 **all-to-all**（dispatch 去 / combine 回）→ 通信模式从「规则的 all-reduce」变成「数据依赖的 all-to-all」→ M4/M5/M6 里多篇网络论文的主角。
  5. 型谱：Mixtral-8x7B（8 专家 top-2）、DeepSeek-V3（256 路由专家 top-8 + 1 共享，细粒度专家）、Switch Transformer（top-1 的历史地位）；**active parameters vs total parameters** 的报数规范。
  6. 为什么 MoE 赢了成本账：同激活 FLOPs 下容量更大；代价：显存占用 = 总参数（都得放着）+ 通信复杂。
- **必收术语**：Mixture-of-Experts（MoE）、expert、router/gating network、top-k routing、sparse activation、active/total parameters、load balancing、auxiliary loss、capacity factor、token dropping、dropless、shared expert、fine-grained experts、expert parallelism（预告）、all-to-all（预告）、dense vs sparse model。
- **定量环节**：DeepSeek-V3 记账：总参数 671B（BF16 放置需 1.34 TB 显存 → 至少 17 张 H100 只为放权重）；每 token 激活 37B → decode 每 token FLOPs ≈ 2×37B 而非 2×671B——推理算力省 18×，但显存一点不省。算清这笔「FLOPs 与显存的剪刀差」。
- **图示**：① MoE 层结构图（router→top-k→experts→combine）；② token 被路由到不同 GPU 上专家的示意（预埋 all-to-all 画面）。
- **延伸阅读**：《Mixtral of Experts》（arXiv 2024，结构清晰易读）；DeepSeek-V3 技术报告第 2 节；《Switch Transformers》（JMLR 2022，选读历史）。
- **误区**：「专家 = 领域专家（数学专家/代码专家）」——实际分工是涌现的、常常不可解释；「MoE 省显存」——省的是 FLOPs，显存反而更费；比较模型大小时混报 active/total。

## L18 注意力变体与长上下文

- **文件**：`M2 Transformer与大模型/L18 注意力变体与长上下文.md`
- **定位**：解释「为什么每家的 attention 都长得不一样」——所有变体都是围绕 KV cache 和 $O(S^2)$ 的系统性优化。密集术语扫盲课。
- **前置**：[[L13 自回归生成与KV缓存]]
- **核心问题**：① MQA/GQA/MLA 各自怎么省 KV cache？② FlashAttention 解决什么（和 GQA 是一类问题吗）？③ 线性注意力/SSM 是什么路线？
- **内容要点**：
  1. 复盘问题源头：decode 是 memory-bound，读的就是 KV cache + 权重 → **省 KV cache = 提吞吐**（承接 L13 定量结论）。
  2. KV 头缩减谱系：**MHA**（每头独立 KV）→ **MQA**（全体共享 1 组 KV，Shazeer 2019）→ **GQA**（分组折中，Llama 系标配）→ **MLA**（DeepSeek：把 KV 压缩成低秩 latent 向量，缓存 latent、用时解压——直觉级讲解 + 压缩比数字）。每种给 KV cache 公式系数变化。
  3. 稀疏/局部路线：**sliding window attention**（Mistral）、**attention sink** 一句话（StreamingLLM 现象）、NSA/MoBA 等 2025 新血一句话提名（标注「快速演化区，读论文时再查」）。
  4. **FlashAttention 定性预告**：不改数学、改访存顺序的 exact attention（tiling+online softmax 的一句话直觉）——与上述「改结构」路线正交；细节留 L51。
  5. 长上下文工程：**RoPE scaling/YaRN**（一句话：位置编码外推）+ 长上下文继续训练；context window 的商业竞赛（128K→1M）与真实能力（**needle-in-a-haystack** 测试一句话）。
  6. 换血路线：**linear attention / SSM / Mamba**（状态空间：把历史压缩成固定大小状态 → decode O(1) 显存）；**hybrid 架构**（若干层 attention + 若干层 SSM）；一句话评价：截至 2026 主流 frontier 仍是 Transformer，但 hybrid 在推理成本上的优势让它持续升温。
- **必收术语**：MQA、GQA、num_key_value_heads、MLA、latent/low-rank compression、sliding window attention、attention sink、RoPE scaling、YaRN（认名）、long context、needle-in-a-haystack、FlashAttention（预告级）、linear attention、state space model（SSM）、Mamba、hybrid architecture、KV cache compression。
- **定量环节**：同一设定（70B 级、S=128K、batch=8）下算四种 KV cache：MHA ≈ 336 GB / GQA(8组) ≈ 42 GB / MQA ≈ 5.3 GB / MLA（按 DeepSeek 压缩比，数字**须核实**技术报告，给量级）——一张表看懂「结构选择 = 显存预算」。
- **图示**：① MHA/MQA/GQA/MLA 四联图（KV 头共享关系，经典图式）；② sliding window 可视范围示意。
- **延伸阅读**：GQA 论文《GQA: Training Generalized Multi-Query Transformer Models…》（EMNLP 2023）；DeepSeek-V2 技术报告（MLA 出处，看图即可）；Mamba 论文（选读，看 abstract 与图 1）。
- **误区**：「FlashAttention 是近似注意力」——它是 exact 的，省的是显存读写；「GQA 掉精度很多」——恰当训练下损失很小，是免费午餐级 tradeoff；「context window 越长越好用」——注意「lost in the middle」与成本。

## L19 实践-解剖迷你GPT

- **文件**：`M2 Transformer与大模型/L19 实践-解剖迷你GPT.md`
- **定位**：【实践课】用 nanoGPT 级别的迷你模型把 M2 全部概念变成手感。CPU/Mac 可跑（模型缩到 ~1M 参数）。
- **前置**：[[L12 Transformer全解剖]]、[[L13 自回归生成与KV缓存]]
- **实践目标（读者要亲眼看到）**：① 数出模型参数量与公式一致；② 训练 loss 从 ~4 降下来、生成质量随之肉眼可见变好；③ 实测 prefill 一段长 prompt vs 逐 token decode 的耗时差；④ 开/关 KV cache 的 decode 速度对比（随生成长度增长曲线）；⑤ 调 temperature/top-p 观察输出多样性。
- **内容要点**：
  1. 环境与代码：提供单文件迷你 GPT（结构与 L12 图纸一一对应：RMSNorm/RoPE 可简化为 LayerNorm/learned PE，但要注明差异）；数据用 tiny shakespeare 或中文诗词小语料。
  2. 任务 A：`sum(p.numel())` 对照手算公式逐项核对。
  3. 任务 B：训练 ~10 分钟，每隔 N step 生成样本，观察「胡言乱语→通顺」的过程（配 loss 曲线）。
  4. 任务 C：写朴素生成 vs KV cache 生成两版代码，测 100/500/1000 token 的耗时曲线，验证 $O(S^2)$ vs $O(S)$。
  5. 挑战题：把 MHA 改成 MQA 数参数变化；数 KV cache 张量的实际内存。
- **必收术语**：（复习课；术语卡片改为「本课亲手验证的 8 个公式/现象」清单）
- **延伸阅读**：Karpathy《Let's build GPT: from scratch》视频 + nanoGPT 仓库（本课蓝本）；bbycroft.net/llm 对照参观。
