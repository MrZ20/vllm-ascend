import fs from 'fs';
const res = JSON.parse(fs.readFileSync('d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything/tmp/ua-file-extract-results-12.json','utf8'));

const nodes=[], edges=[];
const N=(o)=>nodes.push(o);
const E=(s,t,type,w)=>edges.push({source:s,target:t,type,direction:'forward',weight:w});

function fname(p){return p.split('/').pop();}
function comp(nel){return nel<50?'simple':nel<=200?'moderate':'complex';}

const meta = {
'distributed/device_communicators/__init__.py':{s:'设备通信器子包的空初始化文件。',t:['barrel','distributed','entry-point']},
'distributed/device_communicators/npu_communicator.py':{s:'封装 NPU 设备通信器，提供基于 HCCL 的 all_to_all 集合通信原语，供分布式张量并行使用。',t:['service','distributed','npu','communication'],
  cls:{NPUCommunicator:{s:'NPU 设备通信器，初始化 HCCL 通信组并实现 all_to_all 操作。',t:['service','communication','npu','collective']}}},
'distributed/device_communicators/pyhccl.py':{s:'Python 层 HCCL 通信器封装，提供 all_reduce、broadcast 等集合通信操作，桥接 torch-npu 的 HCCL 后端。',t:['service','distributed','npu','collective','communication'],
  cls:{PyHcclCommunicator:{s:'封装 HCCL 通信库，实现 all_reduce 与 broadcast 集合通信原语。',t:['service','communication','collective','npu']}}},
'distributed/kv_transfer/__init__.py':{s:'KV 传输子系统入口，向 vLLM 的 KVConnectorFactory 注册 Ascend 定制的各类 KV connector（Mooncake、AscendStore、CPU offload 等）。',t:['entry-point','factory','registration','kv-cache'],
  fn:{register_connector:{s:'向 vLLM KVConnectorFactory 注册所有 Ascend 定制 KV connector 并覆盖上游默认实现。',t:['factory','registration','kv-cache']}}},
'distributed/kv_transfer/ascend_multi_connector.py':{s:'Ascend 版多连接器，聚合多个 KV connector 并统一处理分配后状态更新与请求完成回调。',t:['service','kv-cache','connector','distributed'],
  cls:{AscendMultiConnector:{s:'聚合多个底层 KV connector，统一转发分配状态更新与请求完成事件。',t:['service','connector','kv-cache']}}},
'distributed/kv_transfer/kv_p2p/__init__.py':{s:'点对点 KV 传输子包的空初始化文件。',t:['barrel','kv-cache','entry-point']},
'distributed/kv_transfer/kv_p2p/mooncake_connector.py':{s:'基于 Mooncake 的点对点 KV cache 传输连接器，实现 prefill/decode 解耦下跨节点的 KV cache 发送与接收线程、调度器与 worker。',t:['service','kv-cache','connector','distributed','disaggregation'],
  cls:{
    KVCacheRecvingThread:{s:'KV cache 接收线程，处理跨节点拉取请求、KV cache 重排（含 mamba/混合布局）及完成信号回收。',t:['service','kv-cache','thread','transfer']},
    KVCacheSendingThread:{s:'KV cache 发送线程，管理待传输请求队列并通过 busy-loop 推进传输完成。',t:['service','kv-cache','thread','transfer']},
    KVCacheTaskTracker:{s:'追踪 KV cache 传输任务的进行与完成状态，支持延迟请求与过期回收。',t:['service','kv-cache','tracking']},
    MooncakeConnector:{s:'Mooncake KV connector 主类，向 vLLM 暴露 load/save KV 接口并委派给调度器与 worker。',t:['service','connector','kv-cache']},
    MooncakeConnectorScheduler:{s:'Mooncake 连接器调度侧逻辑，计算新匹配 token 数、构建连接器元数据与请求完成处理。',t:['service','connector','scheduler','kv-cache']},
    MooncakeConnectorWorker:{s:'Mooncake 连接器 worker 侧实现，负责 KV tensor 注册、远端 rank 解析与跨节点拉取元数据构建。',t:['service','connector','worker','kv-cache']},
    SizedDict:{s:'带容量上限的有序字典，超出后淘汰最旧条目。',t:['utility','data-structure']}
  }},
'distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py':{s:'Mooncake 混合 KV cache 传输连接器，在点对点基础上支持滑动窗口裁剪与混合（attention+mamba）KV 布局的传输。',t:['service','kv-cache','connector','distributed','hybrid'],
  cls:{
    KVCacheRecvingThread:{s:'混合布局 KV cache 接收线程，处理跨节点拉取与 KV cache 重排传输。',t:['service','kv-cache','thread','transfer']},
    MooncakeConnectorScheduler:{s:'混合连接器调度逻辑，实现滑动窗口裁剪、prefill 截断与传输 block 计算。',t:['service','connector','scheduler','kv-cache']},
    MooncakeConnectorWorker:{s:'混合连接器 worker 侧，负责 KV cache 注册及远端 rank 解析。',t:['service','connector','worker','kv-cache']},
    MooncakeConnector:{s:'混合 Mooncake KV connector 主类，向 vLLM 暴露 load/save 接口。',t:['service','connector','kv-cache']}
  }},
'distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py':{s:'Mooncake 逐层 KV cache 传输连接器，支持按 transformer 层粒度流水线式发送与接收 KV cache，降低传输延迟。',t:['service','kv-cache','connector','distributed','layerwise'],
  cls:{
    KVCacheSendingLayerThread:{s:'逐层 KV cache 发送线程，按层粒度处理传输请求与传输元数据生成。',t:['service','kv-cache','thread','layerwise']},
    KVCacheRecvingLayerThread:{s:'逐层 KV cache 接收线程，管理完成/失败请求队列与任务状态更新。',t:['service','kv-cache','thread','layerwise']},
    MooncakeLayerwiseConnector:{s:'逐层 Mooncake KV connector 主类，向 vLLM 暴露按层 load/save KV 接口。',t:['service','connector','kv-cache','layerwise']},
    MooncakeLayerwiseConnectorScheduler:{s:'逐层连接器调度侧逻辑，含 metaserver 访问与请求完成处理。',t:['service','connector','scheduler','kv-cache']},
    MooncakeLayerwiseConnectorWorker:{s:'逐层连接器 worker 侧，负责 KV buffer 创建、block id 对齐与逐层传输。',t:['service','connector','worker','kv-cache']},
    SendReqInfo:{s:'发送请求信息封装，跟踪本地 block id、已计算与已传输 token 数。',t:['data-model','kv-cache']}
  }},
'distributed/kv_transfer/kv_pool/__init__.py':{s:'KV pool 子包的空初始化文件。',t:['barrel','kv-cache','entry-point']},
'distributed/kv_transfer/kv_pool/ascend_store/__init__.py':{s:'Ascend store 子包的空初始化文件。',t:['barrel','kv-cache','entry-point']},
'distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py':{s:'AscendStore KV connector，将 KV cache 卸载到外部存储池（Mooncake/Memcache/Yuanrong），并管理 KV 事件聚合与查找服务。',t:['service','kv-cache','connector','storage'],
  cls:{
    AscendStoreConnector:{s:'AscendStore KV connector 主类，向 vLLM 暴露 load/save KV、事件上报及 worker 元数据构建接口。',t:['service','connector','kv-cache','storage']},
    AscendStoreKVEvents:{s:'聚合各 worker 上报的 KV 存储事件并提供清理与计数能力。',t:['service','events','aggregation']},
    LookupKeyServer:{s:'查找键服务端，响应远端对 KV 存储键存在性的查询。',t:['service','lookup','server']}
  }},
'distributed/kv_transfer/kv_pool/ascend_store/backend/__init__.py':{s:'存储后端子包的初始化文件。',t:['barrel','storage','entry-point']},
'distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py':{s:'存储后端抽象基类，定义 register_buffer、exists、put、get 等 KV 存储统一接口。',t:['type-definition','storage','interface'],
  cls:{Backend:{s:'KV 存储后端抽象基类，规定缓冲区注册与存取的统一接口。',t:['type-definition','storage','interface']}}},
'distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py':{s:'基于 Memcache 的 KV 存储后端实现，封装缓冲区注册与 KV block 的存取。',t:['service','storage','backend','kv-cache'],
  cls:{MemcacheBackend:{s:'Memcache KV 存储后端，实现缓冲区延迟注册与 KV block 存取。',t:['service','storage','backend']}}},
'distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py':{s:'基于 Mooncake 分布式存储的 KV 后端实现，支持 SSD 卸载与从环境/文件加载存储配置。',t:['service','storage','backend','kv-cache'],
  cls:{
    MooncakeBackend:{s:'Mooncake KV 存储后端，实现缓冲区注册与 KV block 的存取，支持 SSD 卸载。',t:['service','storage','backend']},
    MooncakeStoreConfig:{s:'Mooncake 存储配置数据类，支持从文件或环境变量加载。',t:['config','storage','data-model']}
  }},
'distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py':{s:'基于 Yuanrong 的 KV 存储后端实现，含键归一化、blob 列表构建与设备就绪检查。',t:['service','storage','backend','kv-cache'],
  cls:{
    YuanrongBackend:{s:'Yuanrong KV 存储后端，实现缓冲区注册与 KV block 存取。',t:['service','storage','backend']},
    YuanrongHelper:{s:'Yuanrong 后端辅助类，处理键归一化与 blob 列表构建。',t:['utility','storage']},
    YuanrongConfig:{s:'Yuanrong 存储配置数据类，从环境变量加载。',t:['config','storage','data-model']}
  }},
'distributed/kv_transfer/kv_pool/ascend_store/config_data.py':{s:'AscendStore 的核心数据结构与键编码逻辑，定义 PoolKey、ReqMeta、token 数据库及缓存族（cache family）推断等。',t:['data-model','kv-cache','type-definition','serialization'],
  cls:{
    ChunkedTokenDatabase:{s:'分块 token 数据库，按 block hash 生成存储键并准备逐层 KV 值，支持 prefill PP 适配。',t:['data-model','kv-cache','hashing']},
    ReqMeta:{s:'请求元数据，封装 block id 与从请求追踪器构造的传输信息。',t:['data-model','kv-cache']},
    RequestTracker:{s:'请求追踪器，记录已分配 block id 并支持增量更新。',t:['data-model','tracking']},
    PoolKey:{s:'存储池键，支持哈希、字符串编码与按层拆分。',t:['data-model','kv-cache','serialization']},
    LayerMultiBlockReqMeta:{s:'逐层多 block 请求元数据封装。',t:['data-model','kv-cache']}
  }},
'distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py':{s:'AscendStore 的 KV 传输线程实现，包含存储/接收线程及逐层发送/接收线程，负责与外部存储后端的 KV 搬运与事件记录。',t:['service','kv-cache','thread','transfer'],
  cls:{
    KVTransferThread:{s:'KV 传输基础线程，处理请求分发、键查找、token 处理与 KV 事件更新。',t:['service','kv-cache','thread']},
    KVCacheStoreSendingThread:{s:'KV 存储发送线程，管理待存储请求、引用计数与完成事件上报。',t:['service','kv-cache','thread','transfer']},
    KVCacheStoreLayerSendingThread:{s:'逐层 KV 存储发送线程，记录层级事件起始并构建存储完成事件。',t:['service','kv-cache','thread','layerwise']},
    KVCacheStoreRecvingThread:{s:'KV 存储接收线程，处理从存储后端拉取 KV 的请求。',t:['service','kv-cache','thread']},
    KVCacheStoreLayerRecvingThread:{s:'逐层 KV 存储接收线程。',t:['service','kv-cache','thread','layerwise']}
  }},
'distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py':{s:'KV 存储池调度器，负责缓存族/块大小推断、滑动窗口裁剪、新匹配 token 计算与连接器元数据构建。',t:['service','scheduler','kv-cache','storage'],
  cls:{
    KVPoolScheduler:{s:'KV 池调度器，推断分组缓存族与块大小、处理混合/mamba/SWA 缓存并构建连接器元数据。',t:['service','scheduler','kv-cache']},
    LookupKeyClient:{s:'查找键客户端，通过 ZMQ 向查找服务端发起 KV 键存在性查询。',t:['service','lookup','client']}
  }},
'distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py':{s:'KV 存储池 worker，运行在每个 NPU rank 上，负责 KV cache 注册、逐层存取、查找命中计算与存储后端交互。',t:['service','worker','kv-cache','storage'],
  cls:{KVPoolWorker:{s:'KV 池 worker，在 NPU rank 上注册 KV cache、执行逐层存取、查找命中索引计算与 KV 事件管理。',t:['service','worker','kv-cache']}}},
'distributed/kv_transfer/kv_pool/cpu_offload/__init__.py':{s:'CPU 卸载子包的空初始化文件。',t:['barrel','kv-cache','entry-point']},
'distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py':{s:'CPU KV cache 管理器，管理 CPU 侧 KV cache 的前缀缓存命中、slot 分配与释放及缓存统计。',t:['service','kv-cache','cpu-offload','cache-management'],
  cls:{
    CPUKVCacheManager:{s:'CPU KV cache 管理器，处理前缀缓存命中查询、slot 分配与释放。',t:['service','kv-cache','cpu-offload']},
    CPUCacheStats:{s:'CPU 缓存统计，记录命中率等前缀缓存指标。',t:['monitoring','metrics','kv-cache']}
  }},
'distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py':{s:'CPU 卸载 KV connector，使用 aclrtMemcpyBatchAsync 与 torch.npu stream 在 NPU 与 CPU 间异步搬运 KV cache。',t:['service','kv-cache','connector','cpu-offload'],
  cls:{
    CPUOffloadingConnector:{s:'CPU 卸载 KV connector 主类，向 vLLM 暴露 NPU 与 CPU 间的 KV load/save 接口。',t:['service','connector','kv-cache','cpu-offload']},
    CPUOffloadingConnectorScheduler:{s:'CPU 卸载连接器调度侧，计算新匹配 token 与构建连接器元数据。',t:['service','connector','scheduler','kv-cache']},
    CPUOffloadingConnectorWorker:{s:'CPU 卸载连接器 worker 侧，启动元数据服务进程并执行异步 KV 搬运。',t:['service','connector','worker','cpu-offload']}
  }},
'distributed/kv_transfer/kv_pool/cpu_offload/metadata.py':{s:'CPU 卸载元数据服务，基于共享内存在独立进程中维护 CPU KV cache 的前缀缓存元数据。',t:['service','kv-cache','cpu-offload','ipc'],
  cls:{
    MetadataServer:{s:'元数据服务进程，基于共享内存初始化 CPU KV cache 并提供逐步服务循环。',t:['service','ipc','kv-cache','shared-memory']},
    MetadataServerProc:{s:'元数据服务进程封装，负责启动并运行 MetadataServer。',t:['service','ipc','process']}
  }}
};

for(const r of res.results){
  const p=r.path;
  const m=meta[p]||{s:fname(p)+' 源文件。',t:['code','distributed','kv-cache']};
  const fileId='file:'+p;
  N({id:fileId,type:'file',name:fname(p),filePath:p,summary:m.s,tags:m.t,complexity:comp(r.nonEmptyLines)});
  const expNames=new Set((r.exports||[]).map(e=>e.name));
  for(const c of (r.classes||[])){
    const cm=(m.cls&&m.cls[c.name])||null;
    if(!cm) continue;
    const id='class:'+p+':'+c.name;
    N({id,type:'class',name:c.name,filePath:p,lineRange:[c.startLine,c.endLine],summary:cm.s,tags:cm.t,complexity:comp(c.endLine-c.startLine)});
    E(fileId,id,'contains',1.0);
    if(expNames.has(c.name)) E(fileId,id,'exports',0.8);
  }
  for(const f of (r.functions||[])){
    const fm=(m.fn&&m.fn[f.name])||null;
    if(!fm) continue;
    const id='function:'+p+':'+f.name;
    N({id,type:'function',name:f.name,filePath:p,lineRange:[f.startLine,f.endLine],summary:fm.s,tags:fm.t,complexity:comp(f.endLine-f.startLine)});
    E(fileId,id,'contains',1.0);
    if(expNames.has(f.name)) E(fileId,id,'exports',0.8);
  }
}

const initId='file:distributed/kv_transfer/__init__.py';
const regTargets=[
  'distributed/kv_transfer/ascend_multi_connector.py',
  'distributed/kv_transfer/kv_p2p/mooncake_connector.py',
  'distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py',
  'distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py',
  'distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py'
];
for(const t of regTargets) E(initId,'file:'+t,'related',0.5);

const backendId='file:distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py';
for(const t of ['memcache_backend.py','mooncake_backend.py','yuanrong_backend.py']){
  E('file:distributed/kv_transfer/kv_pool/ascend_store/backend/'+t,backendId,'related',0.5);
}

fs.writeFileSync('d:/57108/Desktop/code/vllm-ascend/vllm_ascend/.understand-anything/tmp/gen-12-out.json',JSON.stringify({nodes,edges},null,1));
console.log('nodes',nodes.length,'edges',edges.length);
