# 源码逐跳讲解入口

这个文件是源码阅读主入口。前面的 `chapters/000-015` 按概念讲；这里按代码运行顺序讲，也就是你真正打开源码时应该怎么一步一步跳。

## 教学分支

已经创建两个同名教学分支：

```text
/Users/lonng/Mrz20/vllm-ascend  -> teach-qwen3-5-35b-a3b-source-walkthrough
/Users/lonng/Mrz20/vllm         -> teach-qwen3-5-35b-a3b-source-walkthrough
```

教学文件统一放在 `vllm-ascend/qwen3_5_35b_a3b`。隔壁 `vllm` 分支作为上游源码参照，不额外写教学文件，避免污染上游源码树。

## 为什么不直接在源码里大量加注释

这个阶段先不改生产源码，只写外部源码讲解文档。原因是：

- 讲解内容会很长，放进源码会打断正常 review。
- 同一段源码需要从“小白视角”和“专业视角”重复解释，外部文档更适合。
- 以后如果要做“带注释源码版”，可以在教学分支上再加小范围注释，不影响主线。

## 阅读顺序

1. `source-walkthrough/000-branch-and-reading-map.md`
   建立两个仓库、两个分支、两条源码线的关系。
2. `source-walkthrough/001-test-to-offline-engine.md`
   从 pytest 文件读到 `VllmRunner`，再进入上游 `LLM.generate()` 和 offline engine。
3. `source-walkthrough/002-engine-core-to-npu-runner.md`
   从 `LLMEngine.add_request()`、`EngineCore.step()` 读到 `NPUModelRunner.execute_model()`。
4. `source-walkthrough/003-npu-runner-forward-state.md`
   逐行拆 `execute_model()`：scheduler output、input prep、padding、metadata、forward、logits。
5. `source-walkthrough/004-qwen35-forward-layer.md`
   进入 Qwen3.5 模型层：RMSNorm、full attention、linear attention、MoE。
6. `source-walkthrough/005-fusedmoe-w8a8-source-flow.md`
   进入 MoE 和 W8A8：router、top-k、dispatch、grouped matmul、combine。
7. `source-walkthrough/006-sampling-and-output.md`
   从 logits 到 greedy token，再回到 `RequestOutput`。

## 本轮源码讲解的核心链路

```text
test_qwen3_5_35b_a3b_w8a8.py
  -> VllmRunner.generate_greedy()
  -> VllmRunner.generate()
  -> LLM.generate()
  -> OfflineInferenceMixin._run_completion()
  -> _render_and_add_requests()
  -> LLMEngine.add_request()
  -> OfflineInferenceMixin._run_engine()
  -> LLMEngine.step()
  -> EngineCore.step()
  -> scheduler.schedule()
  -> model_executor.execute_model()
  -> NPUModelRunner.execute_model()
  -> _prepare_inputs()
  -> _determine_batch_execution_and_padding()
  -> _build_attention_metadata()
  -> _preprocess()
  -> set_ascend_forward_context()
  -> _model_forward()
  -> Qwen3_5Model / patched Qwen3_5DecoderLayer.forward()
  -> Qwen3NextSparseMoeBlock.forward()
  -> Ascend FusedMoE / W8A8_DYNAMIC
  -> compute_logits()
  -> NPUModelRunner.sample_tokens()
  -> AscendSampler / vLLM sampler argmax
  -> scheduler.update_from_output()
  -> RequestOutput
```

## 读源码时的纪律

每次看到一个函数，先问四个问题：

```text
1. 这个函数的输入是什么？
2. 它改了哪些状态？
3. 它产出什么给下一个函数？
4. 这一步属于语义计算，还是运行时调度/加速？
```

例如 `num_tokens_padded=8` 属于运行时形状整理；`prompt_token_ids=[9419, 11, 821, 803, 369]` 才属于语义输入。
