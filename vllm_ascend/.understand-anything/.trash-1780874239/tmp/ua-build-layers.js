"use strict";
const fs = require("fs");
const root = "d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything";
const r = JSON.parse(fs.readFileSync(root + "/tmp/ua-arch-results.json", "utf8"));
const dg = r.directoryGroups;

// Map each directory group -> layer id. Consolidated to <=10 layers.
const groupToLayer = {
  "(root)": "platform",
  "device": "platform",
  "device_allocator": "platform",
  "profiler": "platform",

  "attention": "compute-kernels",
  "ops": "compute-kernels",
  "_cann_ops_custom": "compute-kernels",
  "compilation": "compute-kernels",

  "quantization": "quantization",

  "distributed": "distributed",
  "eplb": "distributed",

  "worker": "execution",
  "core": "execution",
  "sample": "execution",
  "spec_decode": "execution",
  "lora": "execution",

  "model_loader": "model",
  "models": "model",

  "kv_offload": "memory",
  "simple_kv_offload": "memory",

  "patch": "patch",

  "_310p": "ascend310p",

  "xlite": "xlite",
};

const layerMeta = {
  platform: {
    id: "layer:platform",
    name: "平台与配置层",
    description: "NPUPlatform 插件入口、ascend_config / envs 全局配置、forward context、设备管理与 profiler，是 vLLM-Ascend 接入 vLLM 并初始化 Ascend NPU 运行环境的基础。",
  },
  "compute-kernels": {
    id: "layer:compute-kernels",
    name: "算子与计算内核层",
    description: "attention 各类后端（MLA/DSA/SFA/FA3）、自定义 NPU 算子与 fused MoE / Triton kernel，以及 torch.compile 融合 pass 与 ACL graph，承担核心数值计算与图优化。",
  },
  quantization: {
    id: "layer:quantization",
    name: "量化层",
    description: "W4A4/W8A8/FP8/MXFP 等量化方案及 compressed-tensors、modelslim 适配，按 layer 类型分发量化 scheme 以在 NPU 上降低显存与提升吞吐。",
  },
  distributed: {
    id: "layer:distributed",
    name: "分布式与并行层",
    description: "并行状态、HCCL 通信器、KV 传输/卸载连接器以及 EPLB 专家并行负载均衡，负责多卡多机的并行通信与专家分布调度。",
  },
  execution: {
    id: "layer:execution",
    name: "调度与执行层",
    description: "worker / Model Runner（含 V2）、core 调度器、sampler、投机解码（spec decode）与 LoRA，驱动每一步推理的请求调度、前向执行与采样。",
  },
  model: {
    id: "layer:model",
    name: "模型与权重加载层",
    description: "elastic/netloader/RFork 等权重加载机制与 DeepSeek V4 等模型定义，负责模型结构构建与权重的弹性加载。",
  },
  memory: {
    id: "layer:memory",
    name: "KV 缓存与卸载层",
    description: "KV cache 卸载（kv_offload / simple_kv_offload）相关机制，管理 KV cache 在 NPU 与主机内存之间的分层存放与迁移。",
  },
  patch: {
    id: "layer:patch",
    name: "运行时补丁层",
    description: "对上游 vLLM platform 与 worker 行为的运行时 monkey-patch，使原生 vLLM 逻辑适配 Ascend NPU 的特性与限制。",
  },
  ascend310p: {
    id: "layer:ascend-310p",
    name: "Ascend 310P 适配层",
    description: "针对 Ascend 310P 设备的并行子树，包含其专属的 attention、ops、fused MoE、quantization、model runner 与权重加载变体。",
  },
  xlite: {
    id: "layer:xlite",
    name: "xLite 轻量运行层",
    description: "xLite 轻量级 runner，提供精简的模型执行路径以支持轻量化部署场景。",
  },
};

// Assign
const layers = {};
for (const meta of Object.values(layerMeta)) layers[meta.id] = { ...meta, nodeIds: [] };

let assigned = 0;
const seen = new Set();
for (const [grp, ids] of Object.entries(dg)) {
  const lid = layerMeta[groupToLayer[grp]] && layerMeta[groupToLayer[grp]].id;
  if (!lid) { console.error("UNMAPPED GROUP:", grp); process.exit(1); }
  for (const id of ids) {
    if (seen.has(id)) { console.error("DUP:", id); process.exit(1); }
    seen.add(id);
    layers[lid].nodeIds.push(id);
    assigned++;
  }
}

const arr = Object.values(layers).filter(l => l.nodeIds.length > 0);
console.log("layers:", arr.length, "assigned:", assigned, "totalExpected:", r.fileStats.totalFileNodes);
for (const l of arr) console.log("  ", l.id, l.name, "=>", l.nodeIds.length);
if (assigned !== r.fileStats.totalFileNodes) { console.error("COUNT MISMATCH"); process.exit(1); }

fs.writeFileSync(root + "/intermediate/layers.json", JSON.stringify(arr, null, 2));
console.log("WROTE layers.json");
