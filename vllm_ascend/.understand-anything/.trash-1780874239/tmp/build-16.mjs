import fs from "fs";

const nodes = [];
const edges = [];

function file(path, name, summary, tags, complexity, notes) {
  const n = { id: `file:${path}`, type: "file", name, filePath: path, summary, tags, complexity };
  if (notes) n.languageNotes = notes;
  nodes.push(n);
}
function fn(path, name, range, summary, tags, complexity) {
  nodes.push({ id: `function:${path}:${name}`, type: "function", name, filePath: path, lineRange: range, summary, tags, complexity });
  edges.push({ source: `file:${path}`, target: `function:${path}:${name}`, type: "contains", direction: "forward", weight: 1.0 });
  edges.push({ source: `file:${path}`, target: `function:${path}:${name}`, type: "exports", direction: "forward", weight: 0.8 });
}
function cls(path, name, range, summary, tags, complexity) {
  nodes.push({ id: `class:${path}:${name}`, type: "class", name, filePath: path, lineRange: range, summary, tags, complexity });
  edges.push({ source: `file:${path}`, target: `class:${path}:${name}`, type: "contains", direction: "forward", weight: 1.0 });
  edges.push({ source: `file:${path}`, target: `class:${path}:${name}`, type: "exports", direction: "forward", weight: 0.8 });
}

// ===== ops/rope_dsv4.py =====
file("ops/rope_dsv4.py", "rope_dsv4.py",
  "为 DeepSeek V4 (DSA) 提供复指数旋转位置编码 (RoPE) 实现，包含全局状态注册表与按层缓存 cos/sin 的代理结构。",
  ["rotary-embedding", "npu", "attention", "data-model"], "complex",
  "使用全局单例 RopeGlobalState 维护跨层注册表与运行时缓冲区，forward 调用 torch_npu.npu_rotary_mul。");
fn("ops/rope_dsv4.py", "get_cos_and_sin_dsa", [61, 99],
  "根据 positions 从全局运行时缓冲区查找并填充 cos/sin 张量，返回封装好的 RopeDataProxy。",
  ["rotary-embedding", "cache", "utility"], "moderate");
cls("ops/rope_dsv4.py", "RopeDataProxy", [22, 58],
  "对按层分组的 cos/sin 缓存数据进行惰性索引访问的代理类，按 layer 配置键路由到对应缓冲区。",
  ["data-model", "proxy", "cache"], "moderate");
cls("ops/rope_dsv4.py", "ComplexExpRotaryEmbedding", [102, 230],
  "基于复指数形式的旋转位置编码模块，支持 YaRN 频率校正并通过 NPU kernel 完成 query 旋转。",
  ["rotary-embedding", "npu", "attention"], "complex");

// ===== ops/rotary_embedding.py =====
file("ops/rotary_embedding.py", "rotary_embedding.py",
  "Ascend NPU 旋转位置编码的核心模块，提供标准/YaRN/DeepSeek 缩放/M-RoPE 等多种 RoPE 实现以及 cos/sin 缓存管理工具函数。",
  ["rotary-embedding", "npu", "attention", "data-model"], "complex",
  "通过 torch.ops.vllm 自定义算子与 torch_npu.npu_mrope/npu_rotary_mul 调度 NPU kernel，含多个 out-of-tree forward_oot 覆写。");
fn("ops/rotary_embedding.py", "set_cos_and_sin", [62, 86],
  "根据 vllm 配置预分配并初始化全局 cos/sin 缓存张量，区分是否为视觉语言模型。",
  ["rotary-embedding", "cache", "initialization"], "moderate");
fn("ops/rotary_embedding.py", "update_cos_sin", [129, 146],
  "依据 positions 从合并的 cos_sin 缓存中索引切片并重排，更新当前批次的 cos/sin 张量。",
  ["rotary-embedding", "cache", "utility"], "simple");
fn("ops/rotary_embedding.py", "rope_forward_oot", [153, 213],
  "out-of-tree 的 RoPE 前向实现，支持 neox/非 neox 风格与部分旋转，调用 torch_npu._npu_rotary_embedding。",
  ["rotary-embedding", "npu", "attention"], "complex");
cls("ops/rotary_embedding.py", "AscendRotaryEmbedding", [216, 250],
  "标准 Ascend 旋转位置编码模块，记录 cos/sin 缓存并通过 NPU 自定义算子完成前向。",
  ["rotary-embedding", "npu", "attention"], "moderate");
cls("ops/rotary_embedding.py", "AscendYaRNRotaryEmbedding", [253, 295],
  "YaRN 缩放变体的 Ascend 旋转位置编码，复用父类 forward_oot 并记录缩放后的缓存。",
  ["rotary-embedding", "npu", "attention"], "moderate");
cls("ops/rotary_embedding.py", "AscendDeepseekScalingRotaryEmbedding", [298, 469],
  "DeepSeek YaRN 缩放旋转位置编码，含 mscale 计算、频率校正范围与 cos/sin 缓存构建逻辑。",
  ["rotary-embedding", "npu", "attention"], "complex");
cls("ops/rotary_embedding.py", "AscendMRotaryEmbedding", [472, 554],
  "多模态 M-RoPE 旋转位置编码，支持 triton 路径与 torch_npu.npu_mrope 两种后端。",
  ["rotary-embedding", "npu", "multimodal", "attention"], "complex");
cls("ops/rotary_embedding.py", "AscendApplyRotaryEmb", [557, 590],
  "对外暴露的 apply rotary embedding 模块，拼接 cos/sin 后调用 npu_rotary_mul 完成旋转。",
  ["rotary-embedding", "npu", "attention"], "moderate");

// ===== __init__.py barrels =====
file("ops/triton/__init__.py", "__init__.py",
  "ops/triton Triton kernel 包的初始化文件，标识该目录为 Python 包。",
  ["entry-point", "barrel", "package"], "simple");
file("ops/triton/activation/__init__.py", "__init__.py",
  "ops/triton/activation 激活函数 kernel 子包的初始化文件。",
  ["entry-point", "barrel", "package"], "simple");
file("ops/triton/batch_invariant/__init__.py", "__init__.py",
  "ops/triton/batch_invariant 批不变算子子包的初始化文件。",
  ["entry-point", "barrel", "package"], "simple");
file("ops/triton/fla/__init__.py", "__init__.py",
  "ops/triton/fla (fused linear attention) kernel 子包的初始化文件。",
  ["entry-point", "barrel", "package"], "simple");
file("ops/triton/linearnorm/__init__.py", "__init__.py",
  "ops/triton/linearnorm 线性层融合归一化 kernel 子包的初始化文件。",
  ["entry-point", "barrel", "package"], "simple");
file("ops/triton/mamba/__init__.py", "__init__.py",
  "ops/triton/mamba Mamba 状态空间模型 kernel 子包的初始化文件。",
  ["entry-point", "barrel", "package"], "simple");

// ===== ops/triton/activation/swiglu_quant.py =====
file("ops/triton/activation/swiglu_quant.py", "swiglu_quant.py",
  "提供 MoE 分组 SwiGLU 激活并融合 FP8 量化的 Triton kernel 及其 Python 封装。",
  ["triton", "kernel", "activation", "quantization", "moe"], "moderate",
  "Triton kernel 按 expert 分组处理，融合 sigmoid 门控与逐组动态量化 scale。");
fn("ops/triton/activation/swiglu_quant.py", "_swiglu_quant_kernel", [8, 63],
  "Triton kernel：对 MoE 各 expert 分组执行 SwiGLU 激活并计算量化 scale 后写出 FP8 输出。",
  ["triton", "kernel", "activation", "quantization"], "complex");
fn("ops/triton/activation/swiglu_quant.py", "swiglu_quant", [66, 100],
  "swiglu_quant 的 Python 入口，校验输入并配置网格调度 Triton kernel。",
  ["triton", "activation", "quantization", "api-handler"], "moderate");

// ===== ops/triton/batch_invariant/matmul.py =====
file("ops/triton/batch_invariant/matmul.py", "matmul.py",
  "批不变 (batch-invariant) 矩阵乘法实现，提供 persistent Triton kernel 及 mm/bmm/addmm/linear 的确定性替代算子。",
  ["triton", "kernel", "matmul", "batch-invariant"], "complex",
  "persistent kernel 固定 tile 调度顺序以保证 batch 间数值一致性，供确定性推理使用。");
fn("ops/triton/batch_invariant/matmul.py", "matmul_bias_persistent_kernel", [25, 98],
  "带偏置的 persistent matmul Triton kernel，使用固定网格遍历 tile 累加点积。",
  ["triton", "kernel", "matmul"], "complex");
fn("ops/triton/batch_invariant/matmul.py", "matmul_persistent", [101, 175],
  "matmul persistent kernel 的 Python 封装，处理维度校验、连续化与网格调度。",
  ["triton", "matmul", "api-handler"], "complex");
fn("ops/triton/batch_invariant/matmul.py", "linear_persistent_kernel", [179, 245],
  "线性层的 persistent Triton kernel，按固定块划分计算 x·yᵀ。",
  ["triton", "kernel", "matmul"], "complex");
fn("ops/triton/batch_invariant/matmul.py", "linear_persistent", [248, 350],
  "linear persistent kernel 的 Python 封装，准备张量并以确定性网格调度。",
  ["triton", "matmul", "api-handler"], "complex");
fn("ops/triton/batch_invariant/matmul.py", "bmm_batch_invariant", [357, 371],
  "批不变的批量矩阵乘 (bmm) 实现，逐 batch 调用 persistent matmul。",
  ["triton", "matmul", "batch-invariant"], "moderate");
fn("ops/triton/batch_invariant/matmul.py", "matmul_batch_invariant", [378, 433],
  "批不变 matmul 调度入口，按输入形状分派到 persistent kernel。",
  ["triton", "matmul", "batch-invariant"], "complex");

// ===== ops/triton/batch_invariant/mean.py =====
file("ops/triton/batch_invariant/mean.py", "mean.py",
  "批不变的均值归约算子，提供按维度求均值的 Triton kernel 及确定性替代实现。",
  ["triton", "kernel", "reduction", "batch-invariant"], "moderate");
fn("ops/triton/batch_invariant/mean.py", "mean_kernel", [25, 67],
  "Triton kernel：以固定块大小对指定维度执行确定性均值归约。",
  ["triton", "kernel", "reduction"], "moderate");
fn("ops/triton/batch_invariant/mean.py", "mean_dim", [70, 154],
  "按指定维度求均值的 Python 封装，重排张量并调度 mean_kernel。",
  ["triton", "reduction", "api-handler"], "complex");
fn("ops/triton/batch_invariant/mean.py", "mean_batch_invariant", [157, 173],
  "批不变 mean 算子入口，规范化维度参数后调用 mean_dim。",
  ["triton", "reduction", "batch-invariant"], "moderate");

// ===== ops/triton/batch_invariant/rmsnorm.py =====
file("ops/triton/batch_invariant/rmsnorm.py", "rmsnorm.py",
  "批不变的 RMSNorm 算子，提供确定性的 Triton kernel 及其封装。",
  ["triton", "kernel", "layernorm", "batch-invariant"], "moderate");
fn("ops/triton/batch_invariant/rmsnorm.py", "_rms_norm_kernel", [25, 78],
  "Triton kernel：逐行计算 RMSNorm，确定性归约平方和后乘权重。",
  ["triton", "kernel", "layernorm"], "complex");
fn("ops/triton/batch_invariant/rmsnorm.py", "rms_norm", [81, 132],
  "RMSNorm 的 Python 封装，校验形状并调度 kernel。",
  ["triton", "layernorm", "api-handler"], "complex");
fn("ops/triton/batch_invariant/rmsnorm.py", "rms_norm_batch_invariant", [135, 153],
  "批不变 RMSNorm 算子入口，规范输入后调用 rms_norm。",
  ["triton", "layernorm", "batch-invariant"], "moderate");

// ===== ops/triton/batch_invariant/softmax.py =====
file("ops/triton/batch_invariant/softmax.py", "softmax.py",
  "提供批不变的 softmax 算子入口，封装底层确定性实现。",
  ["triton", "softmax", "batch-invariant"], "simple");

// ===== ops/triton/batch_memcpy.py =====
file("ops/triton/batch_memcpy.py", "batch_memcpy.py",
  "提供批量内存拷贝的 Triton kernel，按指针数组并行搬运多段数据。",
  ["triton", "kernel", "memory", "utility"], "simple");

// ===== ops/triton/bincount.py =====
file("ops/triton/bincount.py", "bincount.py",
  "提供 token 词表分桶计数及掩码生成的 Triton kernel 及封装，用于张量并行下的频率统计。",
  ["triton", "kernel", "utility", "tensor-parallel"], "moderate");
fn("ops/triton/bincount.py", "token_bin_counts_and_mask_kernel", [32, 88],
  "Triton kernel：按序列分块统计 token 在词表区间内的出现计数。",
  ["triton", "kernel", "reduction"], "complex");
fn("ops/triton/bincount.py", "get_token_bin_counts_and_mask_triton", [91, 151],
  "token 分桶计数的 Python 封装，分配输出并调度 kernel。",
  ["triton", "api-handler", "utility"], "complex");

// ===== ops/triton/fla/fused_qkvzba_split_reshape.py =====
file("ops/triton/fla/fused_qkvzba_split_reshape.py", "fused_qkvzba_split_reshape.py",
  "融合线性注意力 (FLA) 中拆分并重排 QKVZBA 混合张量的 Triton kernel 及封装。",
  ["triton", "kernel", "attention", "reshape"], "moderate",
  "用于 gated delta rule 路径，将 mixed_qkvz 与 mixed_ba 拆分重排为独立的 q/k/v/z/b/a 张量。");
fn("ops/triton/fla/fused_qkvzba_split_reshape.py", "fused_qkvzba_split_reshape_cat_kernel", [21, 141],
  "Triton kernel：按头维度拆分 mixed_qkvz/mixed_ba 并重排拼接出独立张量。",
  ["triton", "kernel", "attention", "reshape"], "complex");
fn("ops/triton/fla/fused_qkvzba_split_reshape.py", "fused_qkvzba_split_reshape_cat", [144, 225],
  "拆分重排 kernel 的 Python 封装，准备输出布局并调度网格。",
  ["triton", "attention", "api-handler"], "complex");

// ===== ops/triton/fla/layernorm_guard.py =====
file("ops/triton/fla/layernorm_guard.py", "layernorm_guard.py",
  "FLA 路径下带门控的 LayerNorm/RMSNorm 前向 Triton kernel 与 autograd 函数封装。",
  ["triton", "kernel", "layernorm", "attention"], "moderate",
  "支持 norm-before-gate 与 RMSNorm 两种模式，LayerNormFn 为 torch.autograd.Function。");
fn("ops/triton/fla/layernorm_guard.py", "layer_norm_fwd_kernel", [22, 91],
  "Triton kernel：逐行计算带可选门控/偏置的 LayerNorm 或 RMSNorm 前向。",
  ["triton", "kernel", "layernorm"], "complex");
fn("ops/triton/fla/layernorm_guard.py", "_layer_norm_fwd", [94, 156],
  "LayerNorm 前向的内部封装，分配中间量并调度 kernel。",
  ["triton", "layernorm", "api-handler"], "complex");
cls("ops/triton/fla/layernorm_guard.py", "LayerNormFn", [159, 198],
  "封装门控 LayerNorm 前向的 autograd.Function，对接 Triton kernel。",
  ["triton", "layernorm", "autograd"], "moderate");

// ===== ops/triton/fla/sigmoid_gating.py =====
file("ops/triton/fla/sigmoid_gating.py", "sigmoid_gating.py",
  "实现 gated delta rule 的 sigmoid 门控融合 Triton kernel，支持变长/连续批/推测解码等场景。",
  ["triton", "kernel", "attention", "gated-delta-rule"], "complex",
  "fused recurrent gated delta rule，融合 sigmoid 门控与状态更新，处理多种 batching 与 spec-decoding 标志。");
fn("ops/triton/fla/sigmoid_gating.py", "fused_recurrent_gated_delta_rule_fwd_kernel", [43, 170],
  "Triton kernel：融合循环 gated delta rule 前向，逐 token 更新隐藏状态。",
  ["triton", "kernel", "attention", "gated-delta-rule"], "complex");
fn("ops/triton/fla/sigmoid_gating.py", "fused_sigmoid_gating_delta_rule_update_kernel", [180, 317],
  "Triton kernel：融合 sigmoid 门控与 delta rule 状态更新计算。",
  ["triton", "kernel", "attention", "gated-delta-rule"], "complex");
fn("ops/triton/fla/sigmoid_gating.py", "fused_sigmoid_gating_delta_rule_update", [320, 393],
  "sigmoid 门控 delta rule 更新的 Python 封装，准备状态源/索引并调度 kernel。",
  ["triton", "attention", "api-handler"], "complex");

// ===== ops/triton/fused_gdn_gating.py =====
file("ops/triton/fused_gdn_gating.py", "fused_gdn_gating.py",
  "提供 gated delta net (GDN) 门控融合的 Triton kernel 及 patch 封装，计算门控 g 与 beta 输出。",
  ["triton", "kernel", "gated-delta-rule", "attention"], "moderate");
fn("ops/triton/fused_gdn_gating.py", "fused_gdn_gating_kernel", [14, 56],
  "Triton kernel：融合计算 GDN 门控 g 与 beta，含 softplus 与阈值处理。",
  ["triton", "kernel", "gated-delta-rule"], "moderate");
fn("ops/triton/fused_gdn_gating.py", "fused_gdn_gating_patch", [59, 99],
  "GDN 门控 kernel 的 Python 封装，分配输出并调度网格。",
  ["triton", "gated-delta-rule", "api-handler"], "moderate");

// ===== ops/triton/gdn_chunk_meta.py =====
file("ops/triton/gdn_chunk_meta.py", "gdn_chunk_meta.py",
  "为 GDN chunked 计算在设备端构建 chunk 元数据（计数/偏移/索引）的 Triton kernel 与辅助函数集合。",
  ["triton", "kernel", "gated-delta-rule", "metadata"], "complex",
  "多个小 kernel 配合从 cu_seqlens 推导 chunk 划分元数据，避免主机端同步。");
fn("ops/triton/gdn_chunk_meta.py", "_build_chunk_meta_device_from_seq_lens", [175, 270],
  "在设备端从序列长度构建完整 chunk 元数据集合，串联多个 kernel 调用。",
  ["triton", "gated-delta-rule", "metadata"], "complex");
fn("ops/triton/gdn_chunk_meta.py", "build_chunk_meta_device", [273, 295],
  "对外入口：校验输入并在设备端构建 GDN chunk 元数据。",
  ["triton", "gated-delta-rule", "api-handler"], "moderate");

// ===== ops/triton/layernorm_gated.py =====
file("ops/triton/layernorm_gated.py", "layernorm_gated.py",
  "针对 NPU 优化的单遍带门控 LayerNorm/RMSNorm 前向 Triton kernel 及封装。",
  ["triton", "kernel", "layernorm", "npu"], "moderate",
  "NPU 专用 1-pass kernel，支持 norm-before-gate 与 RMSNorm 变体。");
fn("ops/triton/layernorm_gated.py", "_layer_norm_fwd_1pass_kernel_npu", [16, 100],
  "NPU 单遍 Triton kernel：融合门控的 LayerNorm/RMSNorm 前向计算。",
  ["triton", "kernel", "layernorm", "npu"], "complex");
fn("ops/triton/layernorm_gated.py", "layer_norm_fwd_npu", [103, 168],
  "NPU 门控 LayerNorm 前向的 Python 封装，分配输出并调度 kernel。",
  ["triton", "layernorm", "api-handler", "npu"], "complex");

// ===== ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py =====
file("ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py", "split_qkv_rmsnorm_mrope.py",
  "融合 QKV 拆分、RMSNorm 与多模态 M-RoPE 的 Triton kernel 及自定义算子注册（含 fake 实现）。",
  ["triton", "kernel", "rotary-embedding", "layernorm", "multimodal"], "complex",
  "单 kernel 融合 split-qkv、QK RMSNorm、M-RoPE 旋转与可选 gate，附带 torch fake 注册用于编译。");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py", "split_qkv_rmsnorm_mrope_kernel", [29, 279],
  "Triton kernel：融合 QKV 拆分、QK RMSNorm 与 M-RoPE 旋转，支持部分旋转与交错布局。",
  ["triton", "kernel", "rotary-embedding", "layernorm"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py", "triton_split_qkv_rmsnorm_mrope", [282, 366],
  "融合 split-qkv-rmsnorm-mrope kernel 的 Python 封装，准备输出并调度。",
  ["triton", "rotary-embedding", "api-handler"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py", "triton_split_qkv_rmsnorm_mrope_fake", [369, 418],
  "对应算子的 fake/meta 实现，供 torch 编译期形状推导使用。",
  ["triton", "fake-impl", "compilation"], "moderate");

// ===== ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py =====
file("ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py", "split_qkv_rmsnorm_rope_simt.py",
  "SIMT 风格融合 QKV 拆分、RMSNorm 与 RoPE 的 Triton kernel，含 cos/sin 预计算 kernel 与算子封装。",
  ["triton", "kernel", "rotary-embedding", "layernorm", "simt"], "complex",
  "SIMT 实现先预计算 RoPE cos/sin，再在主 kernel 内完成 split-qkv、RMSNorm 与旋转。");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py", "precompute_rope_cos_sin_kernel", [11, 38],
  "Triton kernel：按 positions 预计算 RoPE 的 cos/sin 缓存切片。",
  ["triton", "kernel", "rotary-embedding"], "moderate");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py", "split_qkv_rmsnorm_rope_simt_kernel", [42, 264],
  "SIMT Triton kernel：融合 QKV 拆分、QK RMSNorm 与 RoPE 旋转。",
  ["triton", "kernel", "rotary-embedding", "layernorm"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py", "split_qkv_rmsnorm_rope_simt_impl", [267, 357],
  "SIMT 融合算子的 Python 封装，预计算 cos/sin 并调度主 kernel。",
  ["triton", "rotary-embedding", "api-handler"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope_simt.py", "split_qkv_rmsnorm_rope_simt_impl_fake", [360, 394],
  "SIMT 融合算子的 fake/meta 实现，供编译期形状推导。",
  ["triton", "fake-impl", "compilation"], "moderate");

// ===== ops/triton/linearnorm/split_qkv_rmsnorm_rope.py =====
file("ops/triton/linearnorm/split_qkv_rmsnorm_rope.py", "split_qkv_rmsnorm_rope.py",
  "融合 QKV 拆分、RMSNorm 与 RoPE 的 Triton kernel 及自定义算子封装（含 fake 实现）。",
  ["triton", "kernel", "rotary-embedding", "layernorm"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope.py", "split_qkv_rmsnorm_rope_kernel", [26, 261],
  "Triton kernel：融合 QKV 拆分、QK RMSNorm 与 RoPE 旋转，支持部分旋转。",
  ["triton", "kernel", "rotary-embedding", "layernorm"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope.py", "split_qkv_rmsnorm_rope_impl", [264, 344],
  "融合 split-qkv-rmsnorm-rope kernel 的 Python 封装，准备输出并调度。",
  ["triton", "rotary-embedding", "api-handler"], "complex");
fn("ops/triton/linearnorm/split_qkv_rmsnorm_rope.py", "split_qkv_rmsnorm_rope_impl_fake", [347, 381],
  "对应算子的 fake/meta 实现，供 torch 编译期形状推导。",
  ["triton", "fake-impl", "compilation"], "moderate");

// ===== ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py =====
file("ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py", "split_qkv_tp_rmsnorm_rope.py",
  "张量并行下融合 QKV 拆分、全局 RMSNorm 与 RoPE 的 Triton kernel 及算子封装（含 fake 实现）。",
  ["triton", "kernel", "rotary-embedding", "layernorm", "tensor-parallel"], "complex",
  "分两阶段：先计算本地 QK 方差，再跨 TP 聚合后施加全局 RMSNorm 与 RoPE。");
fn("ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py", "_split_qkv_and_compute_local_qk_var_kernel", [37, 104],
  "Triton kernel：拆分 QKV 并计算本地 QK 平方和方差，用于 TP 聚合。",
  ["triton", "kernel", "tensor-parallel"], "complex");
fn("ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py", "_apply_global_rmsnorm_kernel", [108, 230],
  "Triton kernel：基于全局方差对 QK 施加 RMSNorm 并应用 RoPE 旋转。",
  ["triton", "kernel", "layernorm", "rotary-embedding"], "complex");
fn("ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py", "split_qkv_tp_rmsnorm_rope_impl", [233, 300],
  "TP 融合算子的 Python 封装，串联本地方差与全局归一化两个 kernel。",
  ["triton", "rotary-embedding", "api-handler", "tensor-parallel"], "complex");
fn("ops/triton/linearnorm/split_qkv_tp_rmsnorm_rope.py", "split_qkv_tp_rmsnorm_rope_impl_fake", [303, 335],
  "TP 融合算子的 fake/meta 实现，供编译期形状推导。",
  ["triton", "fake-impl", "compilation"], "moderate");

// Partition into 2 parts, files sorted alphabetically by path.
const filePaths = [...new Set(nodes.map(n => n.filePath))].sort();
const parts = 2;
const chunkSize = Math.ceil(filePaths.length / parts);
const groups = [];
for (let i = 0; i < parts; i++) groups.push(new Set(filePaths.slice(i * chunkSize, (i + 1) * chunkSize)));

const dir = "d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything/intermediate";
groups.forEach((g, idx) => {
  const partNodes = nodes.filter(n => g.has(n.filePath));
  const ids = new Set(partNodes.map(n => n.id));
  const partEdges = edges.filter(e => ids.has(e.source));
  fs.writeFileSync(`${dir}/batch-16-part-${idx + 1}.json`, JSON.stringify({ nodes: partNodes, edges: partEdges }, null, 2));
  console.log(`part ${idx + 1}: nodes ${partNodes.length}, edges ${partEdges.length}, files ${g.size}`);
});
console.log("total nodes:", nodes.length, "edges:", edges.length);
