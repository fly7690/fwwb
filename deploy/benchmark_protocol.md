# 性能测试协议与测速规范 (Benchmark Protocol)

为了确保本项目的速度指标可信、可复现，且在答辩环节经得起评委追问，所有涉及模型推理速度的评估均需严格遵守本协议。

## 1. 测试环境与配置记录
每次提交 benchmark 结果时，必须在日志或报告中明确记录以下软硬件与超参设置：
* **硬件信息**：运行测试的 GPU 型号（如通过 SSH 远程测试，需记录实际执行计算的显卡型号）。
* **输入分辨率**：模型实际接收的输入尺寸（如 640x640 或 960x960）。
* **Batch Size**：除非特殊说明，默认测速需设置 batch = 1 。
* **阈值设置**：置信度阈值（Conf Threshold）与 NMS 阈值（IoU Threshold）。

## 2. 测速口径定义 (核心指标)
速度评估必须严格区分以下两种口径，并在最终表格中同时报告 ：

* **纯推理 (Pure Inference)**：仅包含模型在 GPU 上的前向传播（Forward）耗时，**绝不含**任何预处理与后处理。
* **端到端 (End-to-End)**：包含系统流水线的完整处理耗时。具体链路必须包含：视频解码/预处理 + 纯推理 + NMS后处理 + 结果绘制与可视化 + 结构化结果导出（含时间戳。

## 3. 计时规范与 GPU 同步
为了确保统计出的 Latency 真实可信，测速脚本的编写需遵循以下规范：
* **强制 GPU 同步**：在记录推理起始和结束时间戳时，必须显式调用 `torch.cuda.synchronize()`（或 TensorRT 的等价同步手段），防止异步执行导致计时虚小。
* **预热 (Warm-up)**：正式测速前必须进行模型预热（推荐空跑 50 帧），以消除显存初始化和 CUDA 上下文创建带来的时间毛刺。
* **统计样本量**：排除预热帧后，建议在固定视频片段或随机生成的 Tensor 上连续测试至少 500-1000 帧求取平均值。

## 4. 统计指标输出
测速脚本运行结束后，需结构化输出以下指标 ：
* **平均耗时 (Average Latency)**：分为纯推理 Avg Latency 与 端到端 Avg Latency。
* **P90 耗时 (P90 Latency)**：90% 的帧在多少毫秒内处理完毕，用于评估系统应对高负载时的抗抖动能力。
* **FPS (Frames Per Second)**：由平均耗时直接换算得出的纯推理 FPS 与 端到端 FPS。

## 5. 日志输出样例 (Log Template)
测速脚本 (`deploy/bench.py`) 的终端输出需保留类似如下样例，并按需存档供角色 C 与角色 E 接入使用：

```text
[Benchmark Report]
GPU: NVIDIA GeForce RTX 4090 | Input Shape: (1, 3, 640, 640) | Batch: 1
Conf_Thresh: 0.25 | NMS_Thresh: 0.45

--- Pure Inference ---
Avg Latency: XX.X ms
P90 Latency: XX.X ms
FPS: XXX.X

--- End-to-End ---
Avg Latency: XX.X ms
P90 Latency: XX.X ms
FPS: XXX.X
GPU Sync Used: True (torch.cuda.synchronize)