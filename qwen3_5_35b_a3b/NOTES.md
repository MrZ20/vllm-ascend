# 备注

- 用户要求中文讲解，除专有名词、函数名、文件名、命令名外尽量不用英文。
- 教学目标是小白可读：先讲现象，再讲数据变化，最后讲源码跳转。
- 必须保留用户给出的真实工作流：`ssh a3-node0`、`ssh node1`、`docker exec -it zsl_m2m_0612_1 bash`、`cd /vllm-workspace/vllm-ascend`、设置环境变量、`npu-smi info`、选择 `ASCEND_RT_VISIBLE_DEVICES`、运行 pytest。
- 本轮已经完成一次基线 pytest 和一次动态追踪运行。
- 追踪脚本必须作为真实 `.py` 文件运行，不能用 `python -`，因为 `spawn` 子进程需要重新加载主程序文件。
- 用户明确要求不能停留在三言两语或类比层面；涉及的技术点要讲清楚公式、源码、张量形状和真实数据变化。
- 深度版讲解格式：先用简单话说明“这一步做什么”，再给专业解释、数学公式、源码位置、本次实测证据。
- 必须明确区分“测试实际验证了什么”和“模型/系统理论上还涉及什么”；本测试只 assert 输出非空，不是精度测试。
- 讲 W8A8 时不能笼统说全模型 int8，要引用 `quant_model_description.json` 中的代表性条目：MoE expert 是 `W8A8_DYNAMIC`，部分 attention、linear attention、lm_head 仍是 `FLOAT`。
