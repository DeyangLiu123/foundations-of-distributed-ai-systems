---
title: 设计稿 · M3 GPU与加速器
tags:
  - course/design
---

# M3 GPU 与加速器 · 设计稿

> 模块目标：建立**硬件量感**。学完读者应能：读懂 GPU 规格表（并识破稀疏算力营销）、用 roofline 判断任何算子是算力受限还是带宽受限、说清一台 8 卡服务器内部的数据通路。深度标定：到「能读系统论文」为止，不教 CUDA 编程。

## L20 GPU体系结构入门

- **文件**：`M3 GPU与加速器/L20 GPU体系结构入门.md`
- **定位**：回答「GPU 为什么快」。建立 SIMT 执行模型的最小心智图：kernel → grid → block → warp → SM。
- **前置**：[[L04 神经网络与前向传播]]（知道 GEMM 是主角即可）
- **核心问题**：① CPU 和 GPU 的设计哲学差在哪？② 一次矩阵乘在 GPU 上是怎么被成千上万线程分掉的？③ 论文里的 kernel、SM、warp、occupancy 都指什么？
- **内容要点**：
  1. 哲学分歧：CPU = 少量强核 + 大缓存 + 分支预测（**latency-oriented**）；GPU = 海量弱核 + 高带宽（**throughput-oriented**）；「一位教授做题 vs 一千个学生做题」类比。
  2. 硬件层级：GPU = 一百多个 **SM**（streaming multiprocessor）；每个 SM 含 CUDA cores、**Tensor Cores**（下课主角）、寄存器堆、shared memory。H100 = 132 个 SM（引用 03 表）。
  3. 软件层级：**kernel**（在 GPU 上跑的函数）→ **grid → thread block → warp（32 线程锁步）→ thread**；block 被调度到 SM 上执行的映射关系。**SIMT** 的含义与 warp divergence 一句话。
  4. **latency hiding**：GPU 不怕单线程慢，靠海量线程轮转掩盖访存延迟 → **occupancy** 概念（活跃 warp 越多越能藏延迟）。
  5. 执行流水：CPU 发射 kernel（launch overhead ~µs 级）→ 异步执行 → **CUDA stream** 与同步点一句话（profiler 里将天天见，L52）。
  6. 世代速览：V100→A100→H100→B200 的演进主线（算力×、显存带宽×、互联×），指出「每代算力涨得比带宽快」——预埋 roofline 课的悬念。
- **必收术语**：SIMT、SM、CUDA core、Tensor Core（预告）、kernel（GPU 语义）、kernel launch、grid/thread block/warp/thread、warp divergence、occupancy、latency hiding、CUDA stream、host/device、latency-oriented vs throughput-oriented。
- **定量环节**：并行度体感：4096×4096 GEMM 有 1.6×10⁷ 个输出元素；H100 132 个 SM × 每 SM 2048 并发线程 ≈ 27 万在飞线程——「不是更快的核，是多得多的核」。再算 kernel launch overhead：~3–5 µs，decode 一步若发射 1000 个小 kernel → launch 就吃掉几 ms（预埋 CUDA graph 动机，L51）。
- **图示**：① CPU vs GPU 芯片面积分配对比图（经典图式）；② kernel→grid→block→warp→SM 映射图。
- **延伸阅读**：《Programming Massively Parallel Processors》第 1–3 章（PMPP，本模块参考书，选读）；NVIDIA H100 白皮书（看架构图与参数表）。
- **误区**：「GPU 单核也很快」——单线程性能远弱于 CPU；「显卡 = 游戏卡」——数据中心 GPU 无显示输出；kernel ≠ OS kernel ≠ 卷积核（回收 L08 的提醒）。

## L21 GPU内存体系

- **文件**：`M3 GPU与加速器/L21 GPU内存体系.md`
- **定位**：建立「搬数据比算数据贵」的世界观。从寄存器到 HBM 到主机内存的层级、容量、带宽数量级，以及 PyTorch 显存报错怎么读。
- **前置**：[[L20 GPU体系结构入门]]
- **核心问题**：① GPU 里有几层存储、各多大多快？② 数据从磁盘到 Tensor Core 要过几道关？③ CUDA OOM 报错信息每个字段什么意思？
- **内容要点**：
  1. 层级金字塔（标容量与带宽，H100 口径）：寄存器（KB 级/SM，最快）→ **shared memory/L1**（百 KB 级/SM，~TB/s×SM 数）→ **L2**（50 MB，~10 TB/s）→ **HBM**（80 GB，3.35 TB/s）→ 主机 DRAM（TB 级，经 PCIe ~64 GB/s）→ NVMe/网络。相邻层带宽差 ~10×，「每往下一层慢一个量级」。
  2. **HBM** 是什么（堆叠显存，为什么贵、为什么是产能瓶颈一句话）；bandwidth 的物理含义：3.35 TB/s ÷ 989 TFLOPS ≈ 每 FLOP 只配得起 0.003 字节 → 数据必须复用（roofline 伏笔）。
  3. host-device 通道：**PCIe** 世代与带宽（03 表）；**pinned memory** 与异步拷贝；**GPUDirect** 家族预告（P2P/RDMA/Storage，详见 L25/L29/L34）；CXL 一句话提名。
  4. PyTorch 显存现实：**caching allocator**（reserved vs allocated 的区别——nvidia-smi 显示的是 reserved！）、**fragmentation**、OOM 报错逐字段解读（tried to allocate / free / reserved），`torch.cuda.memory_summary` 一句话。
  5. 访存效率一句话版：coalesced access（合并访存）与为什么张量内存布局影响速度（不展开 CUDA 优化）。
- **必收术语**：memory hierarchy、register file、shared memory（SRAM）、L2 cache、HBM、memory bandwidth、DRAM、pinned/page-locked memory、host-to-device copy（H2D/D2H）、GPUDirect（预告）、CXL（提名）、caching allocator、reserved vs allocated、fragmentation、OOM、memory coalescing（一句话级）。
- **定量环节**：搬运 vs 计算：把 80 GB 显存完整读一遍需 80/3350 ≈ 24 ms；这 24 ms 里 H100 能做 989T×0.024 ≈ 2.4×10¹³ FLOPs。反过来：一个 8B 模型 decode 一个 token 至少读 16 GB 权重 → ≥4.8 ms → 上限 ~209 token/s——**第一次从硬件推出 decode 速度上限**（L55 完整版）。
- **图示**：① 存储金字塔（标数字，本模块核心图之一）；② 一个 batch 数据从 SSD→DRAM→HBM→SM 的旅程图。
- **延伸阅读**：NVIDIA CUDA C++ Programming Guide 的 memory hierarchy 节（速览）；PyTorch 文档《CUDA memory management》。
- **误区**：nvidia-smi 显存占用 = 模型大小（是 allocator 的 reserved）；「显存带宽不重要，算力才重要」——LLM 推理恰恰相反；「加 swap 就能解决 OOM」——GPU 显存没有透明 swap（offload 是显式的，L50）。

## L22 算力度量与MFU

- **文件**：`M3 GPU与加速器/L22 算力度量与MFU.md`
- **定位**：教会读者「读规格表 + 算利用率」。揭穿两大行业迷雾：2:4 稀疏营销算力、GPU utilization ≠ 真实利用率。MFU 在此正式定义（全课程反复使用）。
- **前置**：[[L20 GPU体系结构入门]]、[[L16 Scaling Law与算力账]]
- **核心问题**：① H100 的「1979 TFLOPS」为什么要打对折读？② GEMM 为什么能接近峰值而其他算子不能？③ MFU 怎么算、多少算好？
- **内容要点**：
  1. **Tensor Core**：专做小块矩阵乘累加（MMA）的硬件单元；与 CUDA core 的分工；每代支持的 dtype 扩展（V100 FP16 → A100 BF16/TF32 → H100 FP8 → B200 FP4）。
  2. 规格表识读（用 03 表）：按 dtype 的峰值阶梯（FP32 << TF32 << BF16 << FP8）；**2:4 structured sparsity** 的 ×2 营销值——论文与本课程一律用 **dense** 值；「同一张卡在不同精度下是不同的机器」。
  3. **GEMM 效率**：大而方的 GEMM 可达峰值 80–95%；小/瘦 GEMM（decode 的 batch=1 投影）远低于峰值——为什么，留给 L23 roofline 正式解释，这里给现象。
  4. **MFU 定义**（Model FLOPs Utilization）：有效模型 FLOPs ÷ 峰值算力；与 **HFU** 的区别（HFU 把重计算也算进去，MFU 不算——所以 MFU 更「诚实」）；行业参考值：优秀的大规模训练 ~35–45%，MoE 更低。
  5. 揭穿 **GPU utilization**（nvidia-smi 的 Util 列）：只表示「有 kernel 在跑」，util=100% 可能 MFU=5%；正确姿势是算 MFU 或看 profiler（L52）。
  6. 「峰值为什么达不到」原因清单预览：访存受限、通信暴露、kernel launch、气泡——分别指向 L23/L52/L51/L44，本模块只递地图。
- **必收术语**：Tensor Core、MMA、peak FLOPS、dense vs structured sparsity（2:4）、TF32、GEMM efficiency、MFU、HFU、GPU utilization（nvidia-smi 语义）、SOL（speed-of-light，提名）、arithmetic throughput。
- **定量环节**：MFU 三步算：某训练报告称 8B 模型、1024×H100、吞吐 4×10⁶ tokens/s → 4×10⁶×6×8×10⁹ = 1.92×10¹⁷ FLOPs/s ÷（1024×989×10¹²）≈ **19%** → 结论：该系统还有大改进空间。让读者带走这个「审计模板」。
- **图示**：① 规格表阶梯图（同卡不同 dtype 峰值）；② 「util 100% 但 MFU 19%」的示意 timeline（大量空隙被小 kernel 填满）。
- **延伸阅读**：PaLM 论文附录（MFU 定义出处）；NVIDIA H100 数据手册（练习：找出 dense/sparse 两列）。
- **误区**：把稀疏峰值当分母或分子；用 HFU 冒充 MFU（重计算灌水）；「util 高就是没问题」。

## L23 Roofline模型

- **文件**：`M3 GPU与加速器/L23 Roofline模型.md`
- **定位**：全课程最重要的**分析工具**课。一张图统一解释：为什么 decode 慢、为什么要 kernel fusion、为什么量化能加速、为什么 batch 越大越划算。后续至少 6 节课（L51/L55/L58/L59）直接引用本课结论。
- **前置**：[[L21 GPU内存体系]]、[[L22 算力度量与MFU]]
- **核心问题**：① 一个算子的速度上限由什么决定？② arithmetic intensity 怎么算？③ 怎么用 roofline 一眼判断「该优化访存还是算力」？
- **内容要点**：
  1. 两个上限：算力屋顶（peak FLOPS）与带宽斜坡（bandwidth × AI）；**arithmetic intensity（AI）= FLOPs ÷ 访存字节数**；**ridge point** = 峰值算力/带宽——H100 BF16：989T/3.35T ≈ **295 FLOPs/Byte**（记住这个数）。
  2. roofline 图的画法与读法：横轴 AI（对数）、纵轴可达 FLOPS；算子落点在斜坡上 = **memory-bound**，在屋顶下 = **compute-bound**。
  3. 三个标本算 AI：① 逐元素加法（AI≈0.08，深度 memory-bound）；② 方 GEMM n×n×n：AI≈n/3（BF16）→ n=4096 时 ~1365，compute-bound；③ 瘦 GEMM（decode 的 [1,d]×[d,d]）：AI≈1 → 惨烈 memory-bound。**batch 是 AI 的放大器**：[B,d]×[d,d] 的 AI≈B（B 不大时）→「攒 batch = 把算子往屋顶推」——M8 的调度哲学全在这句话里。
  4. 用 roofline 秒答四问：decode 为什么慢（AI≈1）；kernel fusion 为什么有效（省中间读写 → 提 AI，L51）；权重量化为什么加速 decode（分母字节减半 → AI 翻倍，L58）；投机解码为什么有利可图（把串行 decode 变成并行 verify → 提 AI，L59）。每问两三句，埋链接。
  5. 局限一句话：roofline 是上限模型，没算 launch 开销、依赖链、通信——真实系统还要看 profiler（L52）。
- **必收术语**：roofline model、arithmetic intensity（operational intensity）、ridge point、compute-bound、memory-bound、bandwidth-bound、data reuse、operator/op、elementwise operation、fused kernel（预告）。
- **定量环节**：上面三个标本的完整计算（数字进正文）；再出一道：S=8192 prefill 的 attention（大 GEMM 簇）vs 单 token decode 的 attention（读整个 KV cache 只为算一行）各自的 AI 与瓶颈判断——为 L55「prefill compute-bound / decode memory-bound」提前给出证明。
- **图示**：① roofline 主图（标 ridge point 与三个标本落点）——**全课程复用**；② 「batch 把点往右推」的动画式分解图。
- **延伸阅读**：《Roofline: An Insightful Visual Performance Model》（CACM 2009，原始论文，只看图）；《Making Deep Learning Go Brrrr From First Principles》（Horace He 博客，强烈推荐）。
- **误区**：「优化就是换更快的卡」——memory-bound 算子换算力更强的卡毫无收益；「FLOPs 少 = 快」——AI 低的省 FLOPs 算子可能更慢；把 AI 和 MFU 混为一谈（前者是算子属性，后者是系统结果）。

## L24 数值格式

- **文件**：`M3 GPU与加速器/L24 数值格式.md`
- **定位**：dtype 字典课。指数位/尾数位的一次性讲清，建立「精度是可以花的预算」观念。训练侧应用在 L49、推理侧在 L58，本课只管格式本身。
- **前置**：[[L22 算力度量与MFU]]
- **核心问题**：① 浮点数的表示范围和精度由什么决定？② BF16 为什么在训练中赢了 FP16？③ FP8/INT8/INT4 各自的适用场景？
- **内容要点**：
  1. 浮点解剖：sign/exponent/mantissa；**dynamic range**（指数位管）vs **precision**（尾数位管）；用「科学计数法」类比。
  2. 格式对照表（本课主表）：FP32（8e+23m）/ TF32（8e+10m，A100 起的计算模式）/ FP16（5e+10m）/ BF16（8e+7m）/ FP8 两变体（**E4M3** 精度型 / **E5M2** 范围型）/ INT8 / INT4 / 提名 FP4 与 microscaling（**MXFP**，block-wise 共享 scale，B200 世代）。每格式标：字节数、动态范围量级、硬件峰值（引 03 表）。
  3. **BF16 vs FP16 的故事**：FP16 范围窄（最大 65504）→ 梯度/loss 溢出 → 需要 loss scaling 补丁（L49 细讲）；BF16 与 FP32 同指数位 → 范围同、免补丁 → 成为训练默认。「宁牺牲精度不牺牲范围」的工程哲学。
  4. 整数量化入门：scale（+zero point）把浮点映射到整数格；**per-tensor / per-channel / per-group granularity**；outlier 问题一句话（LLM 激活有极端值 → 这是量化研究的核心难点，L58 展开）。
  5. 低精度的收益三连：显存减半、带宽减半（→memory-bound 算子直接提速，回收 L23）、Tensor Core 峰值翻倍——「格式即性能」。
  6. 数值事故谱系一句话版：overflow→Inf、underflow→0、舍入误差累积；为什么 accumulate 常用 FP32（GEMM 内部累加精度）。
- **必收术语**：floating point、sign/exponent/mantissa、dynamic range、precision、FP32/TF32/FP16/BF16、FP8（E4M3/E5M2）、INT8/INT4、MXFP/microscaling（提名）、quantization（预告）、scale factor、per-tensor/per-channel/per-group、outlier（预告）、overflow/underflow、accumulation precision。
- **定量环节**：① 数格子：FP16 在 [1024, 2048) 区间的可表示数间隔（=1）vs BF16（=8）——「BF16 更糙但更皮实」量化呈现；② 8B 模型四种格式的权重体积与 decode 理论上限（回收 L21 公式）：FP32 32GB/105 tok·s⁻¹ → BF16 16GB/209 → FP8 8GB/419 → INT4 4GB/838——「量化 = 免费提速」的 roofline 证明。
- **图示**：① 位域分解对比图（FP32/FP16/BF16/FP8 四条并排，标 e/m）；② 数轴上可表示点密度示意（范围 vs 精度）。
- **延伸阅读**：《Mixed Precision Training》（ICLR 2018，FP16 时代的经典，预告 L49）；NVIDIA Transformer Engine 文档的 FP8 入门节。
- **误区**：「BF16 和 FP16 一样」——字节数同、性格完全不同；「INT8 一定掉精度」——权重量化在恰当校准下几乎无损（L58）；「精度越高越好」——是花预算，高精度买不来更快也未必买来更准。

## L25 节点内互联

- **文件**：`M3 GPU与加速器/L25 节点内互联.md`
- **定位**：打开一台 8 卡服务器的机箱：PCIe / NVLink / NVSwitch 的拓扑与带宽，确立 **scale-up domain** 概念与「TP 不出机箱」的物理依据。它是 M4（跨机网络）的镜像课。
- **前置**：[[L21 GPU内存体系]]
- **核心问题**：① 一台 DGX 里 8 张卡怎么连在一起？② NVLink 比 PCIe/网络快多少、这个差距决定了什么？③ NVL72 为什么被称为「一台机柜大小的 GPU」？
- **内容要点**：
  1. 一台典型 H100 节点解剖（本课主图）：2×CPU、8×GPU、4×NVSwitch、8×400G NIC（每 GPU 一张，**rail** 概念此处首次埋点）、PCIe switch 树、NUMA 域。
  2. **PCIe**：树状、CPU 为根；世代带宽（03 表）；GPU 间走 PCIe 要过 CPU/switch 的代价；**GPUDirect P2P**（GPU 互访显存不经主机内存）。
  3. **NVLink/NVSwitch**：点对点 vs 交换式；世代带宽（03 表：H100 900 GB/s 双向合计）；8 卡经 NVSwitch 全互联 = **all-to-all 无阻塞**；NVLink 上跑的是显存语义（load/store/copy），不是网络包——一句话点破「它更像内存总线」。
  4. 数量级对比（本课灵魂）：HBM 3350 ≫ NVLink 单向 450 ≫ 网卡 50 ≫ PCIe→主机 64（GB/s，H100 口径）——**每跨一级掉一个量级**；由此推出通信密集的并行（TP）只能生活在 NVLink 域内（L43 将定量回收）。
  5. **scale-up domain 的扩张**：GB200 **NVL72**——72 GPU 用 NVLink 连成一柜（NVLink5 1.8 TB/s，NVSwitch tray），「机柜即节点」；对并行策略的含义一句话（TP/EP 可以到 72 了）；提名 **UALink**（开放阵营的 scale-up 标准）与 NVLink Fusion 动向（标注「快速演化区」）。
  6. NUMA/affinity 实务一句话：GPU-NIC-CPU 亲和性对性能的影响（预埋 L38 的 PXN 与 L52 的排障）。
- **必收术语**：PCIe、PCIe switch、NVLink、NVSwitch、bidirectional bandwidth、GPUDirect P2P、NUMA、CPU affinity、DGX/HGX、scale-up vs scale-out（正式定义）、NVLink domain、NVL72、rail（预告）、UALink（提名）、D2D/die-to-die（提名）。
- **定量环节**：搬 16 GB（8B 模型 BF16 权重）各通道耗时：HBM 内 4.8 ms / NVLink 36 ms / 400G 网卡 320 ms / PCIe Gen5 250 ms——一张表建立「数据住在哪」的成本直觉。再算 NVL72 域内 all-to-all 对分带宽与 8 卡节点对比。
- **图示**：① 8 卡 HGX 节点拓扑图（本课主图，M4/M5/M6 反复引用）；② 「带宽悬崖」阶梯图（HBM→NVLink→NIC→PCIe）。
- **延伸阅读**：NVIDIA DGX H100/GB200 NVL72 官方架构页；《How to Scale Your Model》（jax-ml scaling book）的 hardware 章（TPU 视角对照）。
- **误区**：「NVLink 是更快的以太网」——是内存语义互联，走的软件栈完全不同；「8 卡 = 8 倍算力」——通信和显存墙决定了远非线性；PCIe 双向/单向数字混用（回收 03 的口径警告）。

## L26 加速器生态与软件栈

- **文件**：`M3 GPU与加速器/L26 加速器生态与软件栈.md`
- **定位**：模块收尾的「地图课」：GPU 之外的加速器、出口管制下的中国算力现实、以及从 PyTorch 一行代码到硬件指令的完整软件栈。术语扫盲密度高，深度都点到为止。
- **前置**：[[L25 节点内互联]]
- **核心问题**：① TPU 和 GPU 的架构哲学差在哪？② H800/H20 是怎么被「阉割」的、对系统研究意味着什么？③ 从 `torch.matmul` 到 Tensor Core 指令中间隔着什么？
- **内容要点**：
  1. **TPU**：**systolic array**（脉动阵列——数据流过计算单元阵列的直觉图）；**ICI** 互联与 **3D torus** 拓扑（预埋 L31）；**Pod** 概念；XLA 编译器路线 vs CUDA 生态路线的对照（「编译器优先 vs 库优先」）。
  2. 其他玩家一句话画像：AWS Trainium、AMD MI300 系（+ROCm 生态成熟度）、Cerebras/Groq（另类架构，认名即可）。
  3. 中国语境（对本课程读者重要）：出口管制时间线一段话；**H800**（砍 NVLink 到 400 GB/s——通信研究的天然实验场，DeepSeek 论文的大量工程决策源于此）、**H20**（砍算力留带宽——反而适合推理，用 roofline 解释为什么！回收 L23）；国产阵营认名：昇腾 Ascend/CANN、寒武纪等（保持中性事实性，标注「截至 2026」）。
  4. 软件栈纵剖（本课主图）：PyTorch（eager）→ ATen 算子 → 派发到 **cuBLAS/cuDNN**/自定义 kernel → PTX → SASS；旁路：**torch.compile**（图捕获+**Inductor** 生成 **Triton** kernel）；**CUDA Graph**（消 launch 开销，回收 L20 伏笔）；**CUTLASS**（GEMM 模板库，认名）。「90% 的性能问题不需要写 CUDA，需要知道栈里每层在干嘛」。
  5. 生态护城河讨论一段话：CUDA 的真正壁垒是二十年的库+工具+人才；「兼容 CUDA」为什么难（认知即可）。
- **必收术语**：TPU、systolic array、ICI、Pod、XLA、Trainium、ROCm、Ascend/CANN（认名）、export control、H800/H20、CUDA、PTX/SASS（认名）、cuBLAS、cuDNN、CUTLASS（认名）、Triton、torch.compile、Inductor、CUDA Graph、eager mode vs graph mode、operator dispatch。
- **定量环节**：用 03 表对比 H100/H800/H20 三卡：算力比 989:989:148、NVLink 比 900:400:900、显存带宽 3.35:3.35:4.0 TB/s → 推论：H800 训练大模型的痛点在**通信**（所以 DeepSeek 疯狂做通信-计算重叠）、H20 适合**推理**（decode 是 memory-bound）。「从规格表读出系统研究议程」。
- **图示**：① 软件栈纵剖图（PyTorch→…→硬件，标出 torch.compile 旁路）；② systolic array 数据流动画式分解图。
- **延伸阅读**：《TPU v4: An Optically Reconfigurable Supercomputer…》（ISCA 2023，预告 L31 的 OCS）；DeepSeek-V3 技术报告的 infrastructure 节（体会 H800 约束下的工程）；PyTorch 2 论文或 torch.compile 官方博客（选读）。
- **误区**：「国产卡/AMD 卡跑不了大模型」——能跑，差距主要在生态与集群工程成熟度；「编译器能自动解决一切性能问题」——算子级能，分布式策略级（目前）不能；「H20 是废卡」——推理场景反而性价比高（roofline 思维的胜利）。
