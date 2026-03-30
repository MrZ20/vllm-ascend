# Kimi-K2-Thinking

## Introduction

Kimi-K2-Thinking is a large-scale Mixture-of-Experts (MoE) model developed by Moonshot AI. It features a hybrid thinking architecture that excels in complex reasoning and problem-solving tasks.

This document will show the main verification steps of the model, including supported features, environment preparation, single-node deployment, and functional verification.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Kimi-K2-Thinking` (bfloat16): requires 1 Atlas 800 A3 (64G x 16) node. [Download model weight](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

It is recommended to download the model weight to the shared directory, such as `/mnt/sfs_turbo/.cache/`.

### Installation

You can use our official docker image to run `Kimi-K2-Thinking` directly.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image.
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

# Run the container using the defined variables.
# Note: If you are running bridge network with docker, please expose
# available ports for multiple nodes communication in advance.
docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /mnt/sfs_turbo/.cache:/home/cache \
  -it $IMAGE bash
```

## Verify the Quantized Model

Please be advised to edit the value of `"quantization_config.config_groups.group_0.targets"` from `["Linear"]` into `["MoE"]` in `config.json` of original model downloaded from [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

```{code-block} json
{
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "targets": [
          "MoE"
        ]
      }
    }
  }
}
```

Your model files look like:

```{code-block} bash
.
|-- chat_template.jinja
|-- config.json
|-- configuration_deepseek.py
|-- configuration.json
|-- generation_config.json
|-- model-00001-of-000062.safetensors
|-- ...
|-- model-00062-of-000062.safetensors
|-- model.safetensors.index.json
|-- modeling_deepseek.py
|-- tiktoken.model
|-- tokenization_kimi.py
`-- tokenizer_config.json
```

## Online Inference on Multi-NPU

Run the following script to start the vLLM server on Multi-NPU:

For an Atlas 800 A3 (64G*16) node, tensor-parallel-size should be at least 16.

```{code-block} bash
:name: env-preparation
:class: doc-exec
:group: kimi-k2-single-node
export MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-K2-Thinking}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-thinking}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_USE_MODELSCOPE=True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_SMOKE_LOG="${VLLM_SMOKE_LOG:-/tmp/kimi_k2_thinking_smoke.log}"
```

```{code-block} bash
:name: start-service
:class: doc-exec
:group: kimi-k2-single-node
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --tensor-parallel-size 16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 12 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --served-model-name "$SERVED_MODEL_NAME" \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  >"$VLLM_SMOKE_LOG" 2>&1 &
export VLLM_SERVER_PID=$!
echo "Started Kimi-K2-Thinking server with PID $VLLM_SERVER_PID"
```

Once your server is started, you can query the model with input prompts.

```{code-block} bash
:name: verify-service
:class: doc-exec
:group: kimi-k2-single-node
for attempt in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 10
done

curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SERVED_MODEL_NAME"'",
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ],
    "temperature": 1.0,
    "max_tokens": 32
  }'
```

```{code-block} bash
:name: cleanup-service
:class: doc-exec
:group: kimi-k2-single-node
if [ -n "${VLLM_SERVER_PID:-}" ] && kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
  kill "$VLLM_SERVER_PID"
  wait "$VLLM_SERVER_PID" 2>/dev/null || true
fi
```
