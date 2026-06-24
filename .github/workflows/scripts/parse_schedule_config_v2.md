"""
全部先写主逻辑，忽略所有异常处理，后续再加异常处理
1. 重写数据类，去除不必要的参数
@dataclass(frozen=True)
class PeriodicCase:
    name: str
    config_path: str
    test_content: str  # "model" | "accuracy" | "ops"
    framework: str  # "single_node" | "multi_node" | "accuracy" | "ops"
    device_type: str  # "a2" | "a3"  | "310p"
    device_scale: str # "card" | "node"
    device_num: int
    runner: str
    group: str  # schedule_config.yaml中的group字段

2. 重写用例收集流程
(1)不再解析.github/workflows/scripts/runner_label.json，自己做一个映射表
runner_mapping_table
= {
    ("a2", "card", 2): linux-aarch64-a2b3-2,
    ("a2", "node", 2): linux-amd64-cpu-8-hk,
    ("a3", "node", 4): linux-aarch64-a3-0,
    ("a3", "card", 1): linux-aarch64-a3-2,
    ("a3", "card", 2): linux-aarch64-a3-2,
    ("310p", "card", 1): linux-aarch64-310p-2,
    ...
}
(2)for循环遍历所有.github/workflows/scripts/schedule_config.yaml中的路径
创建列表
    single_node_matrix
    multi_node_matrix
    accuracy_matrix
    ops_matrix
for config_path in schedule_config:
    # 路径直接当字符串处理，无需管理文件夹顺序问题，分框架去分组收集PeriodicCase实例
    single_node框架()
        if config_path字符串中含有model，one_node信息
            收集相关参数，创建PeriodicCase实例，补全内容，加入single_node_matrix
            continue
        return
    multi_node框架()
        if config_path字符串中含有model，multi_node信息
            收集相关参数，创建PeriodicCase实例，补全内容，加入multi_node_matrix
            continue
        return
    ops框架()
        if config_path字符串中含有ops信息
            收集相关参数，创建PeriodicCase实例，补全内容，加入ops_matrix
            continue
        return
    accuracy框架()
        if config_path字符串中含有accuracy信息
            收集相关参数，创建PeriodicCase实例，补全内容，加入accuracy_matrix
            continue
        return
    如果走到这里还没有命中，则加入未识别列表，最后打印warning提示
(3) 数据后处理
各种输入参数要求，选择合适的筛选方式，筛选需要测试的用例
比如根据
    group字段筛选
    event-name（schedule）+cron
    test-filter


"""
