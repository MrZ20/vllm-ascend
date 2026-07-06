# 0001 - 建立源码逐跳讲解分支

## 背景

用户希望从已有运行记录继续进入源码级教学，要求讲清楚代码流程、步骤、数据变化和底层原理。

## 本次建立的学习材料

- 创建了 `vllm-ascend` 教学分支：`teach-qwen3-5-35b-a3b-source-walkthrough`
- 创建了隔壁 `vllm` 教学分支：`teach-qwen3-5-35b-a3b-source-walkthrough`
- 新增 `SOURCE_WALKTHROUGH.md` 作为源码逐跳讲解入口。
- 新增 `source-walkthrough/000-006`，按真实调用链讲解从 pytest 到输出的源码路径。

## 当前掌握目标

后续学习不再只问“这个概念是什么”，而要能回答：

```text
这一步对应哪个源码函数？
这个函数的输入是什么？
它改了哪些状态？
它把什么数据交给下一个函数？
它属于语义计算，还是运行时调度/硬件加速？
```

## 下一步建议

下一轮可以选择一个函数做“显微镜阅读”，例如：

```text
NPUModelRunner._prepare_inputs()
NPUModelRunner._build_attention_metadata()
AscendQwen3_5DecoderLayer.forward()
AscendW8A8DynamicFusedMoEMethod.apply()
```
