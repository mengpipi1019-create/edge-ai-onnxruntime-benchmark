# 三端对齐与 CPU 基准

## 数值对齐

先用 Python 的 torchvision 预处理生成 `[1,3,224,224]` float32 张量，分别传给 PyTorch、ORT Python 和 ORT C++。这样可以比较模型计算本身，避免不同图片处理方法影响判断。

五个样本的 Top-5 顺序全部一致。PyTorch 与 ORT Python 的最大 logits 绝对误差为 `8.58306884765625e-06`，阈值为 `1e-4`。ORT Python 与 ORT C++ 的最大误差为 `0`，阈值为 `1e-5`。

第二层让 Python 和 C++ 各自解码 JPEG。五个样本 Top-1 一致，但最大输入差异为 `2.363445`，最大 logits 差异为 `2.414875`。误差在推理前的预处理路径就已出现；JPEG 解码、resize 抗锯齿、坐标和取整规则等差异未逐一消融，不能给出各因素的单独贡献。

结果文件：`results/evidence/2026-08-25_s3_three_runtime_alignment/alignment_results.csv` 与 `alignment_summary.json`。样本来自两张原图及三个确定性变换，只验证一致性，不是准确率评测集。

## 延迟测量

2026-08-25 在 Windows 11、Intel Core i9-14900HX 上固定以下条件：

- ResNet18 FP32，batch=1，相同输入张量。
- 逻辑 CPU 0，框架线程数 1；ORT 使用 CPUExecutionProvider 和顺序执行。
- 每种运行方式 5 个独立进程；每进程 50 次预热和 300 次正式计时。
- 使用固定种子的随机执行顺序，总计 15 个进程、4,500 条计时。
- Python 计时用 `perf_counter_ns`；C++ 用 `steady_clock`，边界仅为一次 forward 或 Session::Run。

主统计量是 5 组进程内延迟中位数的中位数。CV 用 5 个中位数的样本标准差除以均值。PyTorch、ORT Python、ORT C++ 的主延迟分别是 48.13255、29.87945、29.30250 ms；相对于 PyTorch 的加速比分别是 1、1.61089、1.64261。

`benchmark_protocol.json`、`run_schedule.json`、`raw_latency.csv`、`workers/*.csv` 和汇总文件均保留原始数值。运行 `python scripts/verify_published_results.py` 可只读重算；它不产生新的性能结论。

## 未覆盖内容

未测端到端耗时、加载耗时、并发吞吐、量化、内存或功耗，也未测试 Linux 或真实端侧设备。吞吐字段是单请求主延迟的倒数，不是并发压测结果。
