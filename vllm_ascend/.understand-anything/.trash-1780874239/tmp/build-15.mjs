import fs from 'fs';
const ROOT='d:/57108/Desktop/code/vllm-ascend/vllm_ascend';
const ex=JSON.parse(fs.readFileSync(ROOT+'/.understand-anything/tmp/ua-file-extract-results-15.json','utf8'));
const get=p=>ex.results.find(r=>r.path===p);
const nodes=[], edges=[];
const fileNode=(p,name,summary,tags,complexity,notes)=>{const n={id:'file:'+p,type:'file',name,filePath:p,summary,tags,complexity};if(notes)n.languageNotes=notes;nodes.push(n);};
const sub=(type,p,name,lr,summary,tags,complexity)=>{nodes.push({id:type+':'+p+':'+name,type,name,filePath:p,lineRange:lr,summary,tags,complexity});edges.push({source:'file:'+p,target:type+':'+p+':'+name,type:'contains',direction:'forward',weight:1.0});};
const exp=(type,p,name)=>edges.push({source:'file:'+p,target:type+':'+p+':'+name,type:'exports',direction:'forward',weight:0.8});
const isExp=(p,name)=>(get(p).exports||[]).some(e=>e.name===name);
// helper: add significant fn/cls nodes from extraction with provided summaries/tags map
function addCode(p, fnMap, clsMap){
  const r=get(p);
  (r.functions||[]).forEach(f=>{const n=f.endLine-f.startLine+1;const meta=fnMap[f.name];if((n>=10||isExp(p,f.name))&&meta){sub('function',p,f.name,[f.startLine,f.endLine],meta.s,meta.t,meta.c||(n>80?'complex':n>=40?'moderate':'simple'));if(isExp(p,f.name))exp('function',p,f.name);}});
  (r.classes||[]).forEach(c=>{const n=c.endLine-c.startLine+1;const meta=clsMap[c.name];if(((c.methods||[]).length>=2||n>=20||isExp(p,c.name))&&meta){sub('class',p,c.name,[c.startLine,c.endLine],meta.s,meta.t,meta.c||(n>200?'complex':n>=60?'moderate':'simple'));if(isExp(p,c.name))exp('class',p,c.name);}});
}

// ===== ops/dsa.py =====
fileNode('ops/dsa.py','dsa.py','DeepSeek 稀疏注意力（DSA）在 Ascend NPU 上的实现，封装稀疏 attention 前向、KV cache 构建与 metadata 过滤。',['attention','data-model','npu','sparse-attention'],'complex');
addCode('ops/dsa.py',{
  dsa_forward:{s:'DSA 注意力前向计算的自定义算子入口，调度稀疏 attention kernel。',t:['attention','forward','custom-op']},
  _build_kv_cache:{s:'为稀疏注意力构建并组织 KV cache 张量布局。',t:['kv-cache','attention','npu']},
},{
  AscendDeepseekSparseAttention:{s:'Ascend NPU 上的 DeepSeek 稀疏注意力模块，实现 indexer 选择与稀疏 attention 前向。',t:['attention','sparse-attention','data-model','npu']},
});

// ===== ops/flashcomm2_oshard_manager.py =====
fileNode('ops/flashcomm2_oshard_manager.py','flashcomm2_oshard_manager.py','管理 FlashComm2 O-proj 分片（O-shard）的注册、广播触发与加载后处理，协调跨层的张量并行通信。',['service','communication','tensor-parallel','npu'],'moderate');
addCode('ops/flashcomm2_oshard_manager.py',{},{
  Flashcomm2OShardManager:{s:'FlashComm2 O-shard 管理器，负责注册参与层、触发权重广播并在加载后做后处理。',t:['service','communication','tensor-parallel','singleton']},
});

// ===== ops/fused_moe/__init__.py =====
fileNode('ops/fused_moe/__init__.py','__init__.py','fused_moe 子包的初始化文件（空），标记 MoE 算子包。',['entry-point','barrel','npu'],'simple');

// ===== ops/fused_moe/comm_utils.py =====
fileNode('ops/fused_moe/comm_utils.py','comm_utils.py','MoE 专家并行的通信工具，提供异步 all-to-all 与沿首维 all-gather 的集合通信辅助函数。',['utility','communication','moe','npu'],'moderate');
addCode('ops/fused_moe/comm_utils.py',{
  async_all_to_all:{s:'执行异步 all-to-all 集合通信，用于 MoE token 在专家并行组间的分发。',t:['communication','moe','async']},
  _gather_along_first_dim:{s:'沿张量首维执行 all-gather 通信并拼接结果。',t:['communication','all-gather','utility']},
},{});

// ===== ops/fused_moe/experts_selector.py =====
fileNode('ops/fused_moe/experts_selector.py','experts_selector.py','MoE 专家路由与 top-k 选择逻辑，包含分组 top-k、融合算子选择及原生回退实现。',['moe','routing','validation','npu'],'complex');
addCode('ops/fused_moe/experts_selector.py',{
  select_experts:{s:'MoE 路由主入口，根据门控 logits 选择 top-k 专家并计算权重。',t:['moe','routing','top-k'],c:'complex'},
  check_npu_moe_gating_top_k:{s:'校验并适配 NPU 融合 top-k 门控算子的可用条件。',t:['moe','validation','npu']},
  _native_grouped_topk:{s:'分组 top-k 选择的原生 PyTorch 回退实现。',t:['moe','top-k','routing']},
  _select_expert_use_group_topk:{s:'基于专家分组的 top-k 专家选择实现。',t:['moe','routing','top-k']},
  _select_experts_with_fusion_ops:{s:'使用 NPU 融合算子完成专家选择与权重归一化。',t:['moe','routing','npu'],c:'moderate'},
  _native_select_experts:{s:'专家选择的原生实现，覆盖 softmax/sigmoid 评分与归一化路径。',t:['moe','routing','top-k'],c:'complex'},
  zero_experts_compute:{s:'处理 zero-expert（占位专家）的权重计算。',t:['moe','routing']},
},{});

// ===== ops/fused_moe/fused_moe.py =====
fileNode('ops/fused_moe/fused_moe.py','fused_moe.py','Ascend 融合 MoE 层的核心实现，提供非量化 MoE 方法、运行器与 AscendFusedMoE 模块，串联门控、专家计算与共享专家。',['moe','data-model','npu','service'],'complex','文件较大（800+ 行），围绕 FusedMoE 层组织门控/分发/专家/共享专家的完整前向流水线。');
addCode('ops/fused_moe/fused_moe.py',{},{
  AscendUnquantizedFusedMoEMethod:{s:'非量化 MoE 计算方法，负责权重加载后处理与 apply 前向调度。',t:['moe','method','npu'],c:'moderate'},
  AscendMoERunner:{s:'MoE 前向运行器，处理 DP 分块、输出 reduce 与共享专家输出合并。',t:['moe','runtime','data-parallel'],c:'moderate'},
  AscendFusedMoE:{s:'Ascend 融合 MoE 模块主类，整合门控、专家映射、共享专家与张量并行 all-reduce 的完整前向。',t:['moe','data-model','npu','service'],c:'complex'},
});

// ===== ops/fused_moe/gate_linear.py =====
fileNode('ops/fused_moe/gate_linear.py','gate_linear.py','MoE 门控线性层封装，实现路由 logits 的线性投影。',['moe','gate','data-model','npu'],'simple');
addCode('ops/fused_moe/gate_linear.py',{},{
  AscendGateLinear:{s:'MoE 门控线性层，对隐藏态做线性投影产生路由 logits。',t:['moe','gate','linear']},
});

// ===== ops/fused_moe/moe_comm_method.py =====
fileNode('ops/fused_moe/moe_comm_method.py','moe_comm_method.py','MoE 通信方法抽象与多种实现（AllGather/MC2/AllToAll/FusedMC2），统一 prepare/finalize 与专家计算的通信策略。',['moe','communication','factory','npu'],'complex');
addCode('ops/fused_moe/moe_comm_method.py',{},{
  MoECommMethod:{s:'MoE 通信方法抽象基类，定义 prepare/finalize/fused_experts 与 token dispatcher 接入点。',t:['moe','communication','abstract'],c:'moderate'},
  AllGatherCommImpl:{s:'基于 all-gather 的 MoE 通信实现。',t:['moe','communication','all-gather']},
  MC2CommImpl:{s:'基于 MC2 算子的 MoE 通信实现。',t:['moe','communication','mc2']},
  AlltoAllCommImpl:{s:'基于 all-to-all 的 MoE 通信实现。',t:['moe','communication','all-to-all']},
  FusedMC2CommImpl:{s:'融合 MC2 的 MoE 通信实现，将专家计算与通信融合以降低开销。',t:['moe','communication','mc2','npu'],c:'moderate'},
});

// ===== ops/fused_moe/moe_mlp.py =====
fileNode('ops/fused_moe/moe_mlp.py','moe_mlp.py','MoE 专家 MLP 计算核心，提供量化/非量化的 group GEMM + SwiGLU 实现与统一调度入口。',['moe','quantization','npu','data-pipeline'],'complex');
addCode('ops/fused_moe/moe_mlp.py',{
  cumsum_group_list:{s:'对分组 token 计数做累加，生成 group GEMM 所需的偏移列表。',t:['moe','utility','npu']},
  quant_apply_mlp:{s:'量化 MoE 专家 MLP 计算，覆盖 FP8/INT 等量化路径的 group GEMM 与 SwiGLU。',t:['moe','quantization','mlp','npu'],c:'complex'},
  unquant_apply_mlp:{s:'非量化 MoE 专家 MLP 计算实现。',t:['moe','mlp','npu'],c:'moderate'},
  unified_apply_mlp:{s:'统一的专家 MLP 调度入口，按量化配置分派到量化或非量化实现。',t:['moe','mlp','dispatch'],c:'moderate'},
},{});

// ===== ops/fused_moe/moe_runtime_args.py =====
fileNode('ops/fused_moe/moe_runtime_args.py','moe_runtime_args.py','构建 MoE 运行时参数的工厂函数集合，组装专家计算、token 分发与 MLP 计算所需的输入结构。',['moe','factory','runtime','npu'],'complex');
addCode('ops/fused_moe/moe_runtime_args.py',{
  _build_mxfp_params:{s:'构建 MXFP 量化相关的运行时参数。',t:['moe','quantization','factory']},
  build_fused_experts_input:{s:'组装融合专家计算所需的完整输入参数结构。',t:['moe','factory','runtime'],c:'moderate'},
  build_token_dispatch_input:{s:'构建 token 分发阶段的输入参数。',t:['moe','factory','runtime']},
  build_mlp_compute_input:{s:'构建专家 MLP 计算阶段的输入参数。',t:['moe','factory','runtime']},
},{});

// ===== ops/fused_moe/moe_stage_contracts.py =====
fileNode('ops/fused_moe/moe_stage_contracts.py','moe_stage_contracts.py','定义 MoE 各阶段间数据契约的 dataclass 集合（prepare 输出、权重、专家输入、分发输入输出、combine metadata 等）。',['moe','type-definition','data-model','npu'],'moderate');
addCode('ops/fused_moe/moe_stage_contracts.py',{},{
  MoEPrepareOutput:{s:'MoE prepare 阶段输出数据契约。',t:['moe','type-definition','data-model']},
  MoEFusedExpertsInput:{s:'融合专家计算输入数据契约。',t:['moe','type-definition','data-model']},
  MoETokenDispatchInput:{s:'token 分发输入数据契约。',t:['moe','type-definition','data-model']},
  MoETokenDispatchOutput:{s:'token 分发输出数据契约。',t:['moe','type-definition','data-model']},
  MoEMlpComputeInput:{s:'专家 MLP 计算输入数据契约。',t:['moe','type-definition','data-model']},
});

// ===== ops/fused_moe/moe_stage_params.py =====
fileNode('ops/fused_moe/moe_stage_params.py','moe_stage_params.py','MoE 量化参数模型，封装是否量化、MXFP/INT 量化与 W4A8 等配置判定。',['moe','quantization','type-definition','npu'],'moderate');
addCode('ops/fused_moe/moe_stage_params.py',{},{
  MoEQuantParams:{s:'MoE 量化参数 dataclass，提供 is_quant/is_mxfp/is_int_quant 等量化模式判定。',t:['moe','quantization','type-definition']},
});

// ===== ops/fused_moe/prepare_finalize.py =====
fileNode('ops/fused_moe/prepare_finalize.py','prepare_finalize.py','MoE prepare/finalize 通信策略的多种实现（All2All/MC2/AllGather），处理 DP/TP 维度的张量重排与 token padding。',['moe','communication','factory','npu'],'complex');
addCode('ops/fused_moe/prepare_finalize.py',{},{
  PrepareAndFinalize:{s:'MoE prepare/finalize 基类，定义无通信的默认 prepare 与 finalize。',t:['moe','communication','abstract'],c:'moderate'},
  PrepareAndFinalizeWithAll2All:{s:'基于 all-to-all 的 prepare/finalize，处理跨 DP 的 TP 还原与 token padding。',t:['moe','communication','all-to-all'],c:'moderate'},
  PrepareAndFinalizeWithMC2:{s:'基于 MC2 的 prepare/finalize 实现。',t:['moe','communication','mc2'],c:'moderate'},
  PrepareAndFinalizeWithAllGather:{s:'基于 all-gather 的 prepare/finalize，覆盖 EP/DP 组的分发与合并路径。',t:['moe','communication','all-gather'],c:'complex'},
});

// ===== ops/fused_moe/token_dispatcher.py =====
fileNode('ops/fused_moe/token_dispatcher.py','token_dispatcher.py','MoE token 分发器抽象与多种实现（MC2/AllGather/All2AllV），负责 token 在专家并行组间的 dispatch 与 combine。',['moe','communication','factory','npu'],'complex');
addCode('ops/fused_moe/token_dispatcher.py',{},{
  MoETokenDispatcher:{s:'token 分发器抽象基类，定义 dispatch/combine 接口与 EP 组属性。',t:['moe','communication','abstract'],c:'moderate'},
  TokenDispatcherWithMC2:{s:'基于 MC2 算子的 token 分发器，构建 dispatch/combine 的 kwargs 并调度。',t:['moe','communication','mc2'],c:'complex'},
  TokenDispatcherWithAllGather:{s:'基于 all-gather 的 token 分发器实现。',t:['moe','communication','all-gather'],c:'moderate'},
  TokenDispatcherWithAll2AllV:{s:'基于 all-to-all-v 的 token 分发器，含 dispatch/combine 的前后处理流水线。',t:['moe','communication','all-to-all'],c:'complex'},
});

// ===== ops/gdn.py =====
fileNode('ops/gdn.py','gdn.py','门控 Delta 网络（Gated Delta Net）注意力在 Ascend NPU 上的实现，含 causal conv1d host args 处理与图模式参数更新。',['attention','data-model','npu','graph-mode'],'complex','文件较大（650+ 行），涉及 ACL graph capture 下 conv1d host 参数的 padding 与图参数更新。');
addCode('ops/gdn.py',{
  _pad_conv1d_host_args_to_capture:{s:'为图捕获（graph capture）对 conv1d 的 host 端参数做 padding 对齐。',t:['attention','graph-mode','npu']},
  update_conv1d_graph_params:{s:'在 ACL graph 模式下更新 conv1d 算子的图参数。',t:['attention','graph-mode','npu'],c:'complex'},
},{
  AscendGatedDeltaNetAttention:{s:'Ascend 门控 Delta 网络注意力模块，实现 chunked prefill、kernel 预热与核心前向。',t:['attention','data-model','npu'],c:'complex'},
});

// ===== ops/layer_shard_linear.py =====
fileNode('ops/layer_shard_linear.py','layer_shard_linear.py','分层权重分片（layer shard）线性层支持，按层序列注册权重分片并在加载后串联处理。',['data-model','tensor-parallel','npu','service'],'complex');
addCode('ops/layer_shard_linear.py',{
  register_layer_to_shard_weight_series:{s:'将单个层注册到权重分片序列中，建立分片的源-目标关系。',t:['tensor-parallel','registration','npu']},
  register_all_layers_to_shard_weight_series:{s:'批量注册所有层到权重分片序列。',t:['tensor-parallel','registration']},
},{
  SeriesMetadata:{s:'权重分片序列元数据，跟踪源层判定、加载后处理与层间权重等待/触达。',t:['tensor-parallel','data-model','npu'],c:'moderate'},
});

// ===== ops/layernorm.py =====
fileNode('ops/layernorm.py','layernorm.py','Ascend NPU 上的 RMSNorm/LayerNorm 算子封装，含标准、Gemma 与门控（gated）变体。',['layernorm','data-model','npu','component'],'moderate');
addCode('ops/layernorm.py',{},{
  AscendRMSNorm:{s:'Ascend RMSNorm 算子，提供 NPU 上的 RMS 归一化前向与偏置权重加载。',t:['layernorm','npu','component']},
  AscendGemmaRMSNorm:{s:'Gemma 风格 RMSNorm 变体的 NPU 实现。',t:['layernorm','npu','component']},
  LayerNormFn:{s:'通用 LayerNorm 前向函数封装。',t:['layernorm','npu','component']},
  AscendRMSNormGated:{s:'带门控的 RMSNorm 变体，实现门控归一化前向。',t:['layernorm','npu','component']},
});

// ===== ops/linear_op.py =====
fileNode('ops/linear_op.py','linear_op.py','自定义并行线性算子层级体系，覆盖列/行并行、序列并行、MLP、O-proj 及 FlashComm2/MatmulAllreduce 等多种通信融合策略。',['data-model','tensor-parallel','npu','factory'],'complex','文件较大（600+ 行），用类层级对不同张量并行通信模式（all-reduce/sequence/sharded-cp）做策略化封装。');
addCode('ops/linear_op.py',{
  get_parallel_op:{s:'并行算子工厂入口，按配置选择列/行并行算子实现。',t:['tensor-parallel','factory','dispatch']},
  is_moe_layer:{s:'判定给定层是否为 MoE 层。',t:['moe','utility','validation']},
  _get_column_parallel_op:{s:'选择合适的列并行算子实现。',t:['tensor-parallel','factory']},
  _get_row_parallel_op:{s:'选择合适的行并行算子实现。',t:['tensor-parallel','factory']},
},{
  CustomLinearOp:{s:'自定义线性算子基类，封装通信组、TP rank/size 与 apply 调度。',t:['tensor-parallel','abstract','npu']},
  CustomColumnParallelOp:{s:'列并行线性算子基类。',t:['tensor-parallel','npu']},
  CustomRowParallelOp:{s:'行并行线性算子基类，含输入并行切分。',t:['tensor-parallel','npu']},
  OProjRowParallelOp:{s:'注意力 O-proj 的行并行算子实现。',t:['tensor-parallel','attention','npu']},
  Flashcomm2OProjRowParallelOp:{s:'基于 FlashComm2 的 O-proj 行并行算子，融合通信优化。',t:['tensor-parallel','communication','npu'],c:'moderate'},
  MatmulAllreduceRowParallelOp:{s:'融合 matmul 与 all-reduce 的行并行算子。',t:['tensor-parallel','communication','npu']},
  SequenceRowParallelOp:{s:'序列并行的行并行算子，实现 matmul 与 reduce-scatter 融合。',t:['tensor-parallel','sequence-parallel','npu'],c:'moderate'},
});

// ===== ops/linear.py =====
fileNode('ops/linear.py','linear.py','Ascend 线性层族，覆盖 QKV/Merged-Column/Row/Column/Replicated 等并行线性层及非量化线性方法。',['data-model','tensor-parallel','npu','component'],'complex');
addCode('ops/linear.py',{},{
  AscendUnquantizedLinearMethod:{s:'非量化线性计算方法，处理权重加载后处理与 apply 前向。',t:['linear','method','npu']},
  AscendLinearBase:{s:'Ascend 线性层基类，封装公共初始化逻辑。',t:['linear','abstract','npu']},
  AscendQKVParallelLinear:{s:'QKV 融合并行线性层的 NPU 实现。',t:['linear','tensor-parallel','attention'],c:'moderate'},
  AscendMergedColumnParallelLinear:{s:'合并列并行线性层实现。',t:['linear','tensor-parallel','npu'],c:'moderate'},
  AscendRowParallelLinear:{s:'行并行线性层实现，含 reduce 通信。',t:['linear','tensor-parallel','npu'],c:'moderate'},
  AscendColumnParallelLinear:{s:'列并行线性层实现，含权重加载器。',t:['linear','tensor-parallel','npu'],c:'complex'},
  AscendReplicatedLinear:{s:'复制（非并行）线性层实现。',t:['linear','npu','component'],c:'moderate'},
});

// ===== ops/mhc.py =====
fileNode('ops/mhc.py','mhc.py','提供 Sinkhorn 归一化的参考实现（hc_split_sinkhorn_ref），用于多头路由/分配相关计算。',['utility','routing','npu'],'simple');
addCode('ops/mhc.py',{
  hc_split_sinkhorn_ref:{s:'Sinkhorn 归一化的参考实现，对分配矩阵做行列归一化迭代。',t:['routing','utility','reference']},
},{});

// ===== ops/mla.py =====
fileNode('ops/mla.py','mla.py','多头潜在注意力（MLA）在 Ascend NPU 上的实现，封装 indexer wrapper 与 MLA 前向自定义算子。',['attention','data-model','npu','mla'],'moderate');
addCode('ops/mla.py',{
  mla_forward:{s:'MLA 注意力前向的自定义算子入口。',t:['attention','forward','custom-op']},
},{
  IndexerWrapper:{s:'MLA indexer 封装，处理潜在向量的索引选择前向。',t:['attention','mla','component']},
  AscendMultiHeadLatentAttention:{s:'Ascend 多头潜在注意力模块，实现 MLA 的完整前向。',t:['attention','mla','data-model','npu'],c:'moderate'},
});

// ===== ops/mm_encoder_attention.py =====
fileNode('ops/mm_encoder_attention.py','mm_encoder_attention.py','多模态编码器注意力（MM encoder attention）的 NPU 实现，处理变长序列的 QKV 重排与 cu_seqlens 计算。',['attention','multimodal','npu','component'],'moderate');
addCode('ops/mm_encoder_attention.py',{},{
  AscendMMEncoderAttention:{s:'多模态编码器注意力模块，含序列长度计算、QKV 三维重排与变长注意力前向。',t:['attention','multimodal','npu'],c:'moderate'},
});

// ===== ops/qwen2_decoder.py =====
fileNode('ops/qwen2_decoder.py','qwen2_decoder.py','Qwen2 解码器在 Ascend 上的优化定制，包含定制 decoder/model/layer/attention/RMSNorm 组件。',['data-model','decoder','npu','component'],'complex');
addCode('ops/qwen2_decoder.py',{},{
  AscendCustomQwen2Decoder:{s:'Qwen2 定制解码器构建器，生成针对 NPU 优化的定制模型。',t:['decoder','npu','factory'],c:'moderate'},
  AscendQwen2Model:{s:'Qwen2 模型的 Ascend 定制实现，组织各层前向。',t:['decoder','model','npu'],c:'moderate'},
  AscendQwen2DecoderLayer:{s:'Qwen2 解码器单层的 NPU 实现。',t:['decoder','npu','component']},
  AscendQwen2Attention:{s:'Qwen2 注意力子层的 NPU 实现。',t:['attention','decoder','npu']},
  AscendQwen2RMSNorm:{s:'Qwen2 RMSNorm 子层的 NPU 实现。',t:['layernorm','decoder','npu']},
});

// ===== ops/register_custom_ops.py =====
fileNode('ops/register_custom_ops.py','register_custom_ops.py','向 torch 自定义算子库注册 Ascend NPU 算子（residual 分块、all-gather、pad-reduce、matmul-reduce、rope 等）及其 fake 实现。',['registration','custom-op','npu','service'],'complex');
addCode('ops/register_custom_ops.py',{
  _maybe_chunk_residual_impl:{s:'residual 张量按需分块的自定义算子实现。',t:['custom-op','npu','registration']},
  _maybe_all_gather_and_maybe_unpad_impl:{s:'按需 all-gather 并去除 padding 的自定义算子实现。',t:['custom-op','communication','npu']},
  _maybe_pad_and_reduce_impl:{s:'按需 padding 并 reduce 的自定义算子实现。',t:['custom-op','communication','npu']},
},{});

// ===== ops/rel_pos_attention.py =====
fileNode('ops/rel_pos_attention.py','rel_pos_attention.py','相对位置编码注意力（relative position attention）的 Ascend NPU 实现。',['attention','npu','component'],'simple');
addCode('ops/rel_pos_attention.py',{},{
  AscendRelPosAttention:{s:'相对位置注意力模块的 NPU 实现，含相对位置偏置的前向计算。',t:['attention','npu','component'],c:'moderate'},
});

// ===== write =====
fs.writeFileSync(ROOT+'/.understand-anything/tmp/ua-15-full.json',JSON.stringify({nodes,edges},null,1));
console.log('nodes',nodes.length,'edges',edges.length);
