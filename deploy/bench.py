import torch
import time
import numpy as np
import pandas as pd
import os

#参数配置（更具具体情况再调整）
CONFIG = {
    "model_path": "yolov8n.pt",
    "input_size": (640, 640),
    "batch_size": 1,
    "conf_thres": 0.25,
    "nms_thres": 0.45,
    "warmup_steps": 50,
    "test_steps": 300
}

#获取硬件信息
def get_gpu_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

#强制GPU同步
def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return True
    return False

@torch.no_grad()
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_info()

    try:
        from ultralytics import YOLO
        model = YOLO(CONFIG["model_path"]).model.to(device)
    except Exception as e:
        print(f"fail to model: {e}")
        return   

    model.eval()
    dummy_input = torch.randn(CONFIG["batch_size"], 3, *CONFIG["input_size"]).to(device)

    #预热
    print(f"warm up {CONFIG['warmup_steps']} steps")
    for _ in range(CONFIG["warmup_steps"]):
        _ = model(dummy_input)
    synchronize()

    #纯推理
    pure_times = []
    for _ in range(CONFIG["test_steps"]):
        synchronize()
        start = time.perf_counter()
        _ = model(dummy_input)
        synchronize()
        pure_times.append((time.perf_counter() - start) * 1000)

    e2e_times = []
    for _ in range(CONFIG["test_steps"]):
        start = time.perf_counter()

        #模拟预处理
        _ = torch.randn(1, 3, *CONFIG["input_size"]).to(device) / 255.0

        #推理
        _ = model(dummy_input)

        #模拟 NMS 与结构化数据处理（固定模型耗时）
        #到V1阶段替换成真实的预处理逻辑
        synchronize()
        time.sleep(0.008)

        e2e_times.append((time.perf_counter() - start) * 1000)

    #数据导出与记录
    stats = {}
    for mode, data in [("Pure Inference", pure_times), ("End-to-End", e2e_times)]:
        stats[mode] = {
            "avg_lat" : np.mean(data),
            "p90_lat" : np.percentile(data, 90),
            "fps" : 1000 / np.mean(data)
        }

    print(f"\n[Benchmark Report]")
    print(f"GPU: {gpu_info} | Input Shape: {CONFIG['input_size']} | Batch: {CONFIG['batch_size']}")
    print(f"Conf_Thresh: {CONFIG['conf_thres']} | NMS_Thresh: {CONFIG['nms_thres']}")

    for mode in ["Pure Inference", "End-to-End"]:
        print(f"\n--- {mode} ---")
        print(f"Avg Latency: {stats[mode]['avg_lat']:.2f} ms")
        print(f"P90 Latency: {stats[mode]['p90_lat']:.2f} ms")
        print(f"FPS: {stats[mode]['fps']:.2f}")

    sync_status = synchronize()
    print(f"GPU Sync Used: {sync_status}\n")

    result = []
    for mode in ["Pure Inference", "End-to-End"]:
        result.append({
            "Metric": mode,
            "GPU": gpu_info,
            "Resolution": f"{CONFIG['input_size'][0]} x {CONFIG['input_size'][1]}",
            "Batch": CONFIG["batch_size"],
            "Conf_Thresh": CONFIG["conf_thres"],
            "NMS_Thresh": CONFIG["nms_thres"],
            "Avg_Latency_ms": round(stats[mode]["avg_lat"], 2),
            "P90_Latency_ms": round(stats[mode]["p90_lat"], 2),
            "FPS": round(stats[mode]["fps"], 2),
            "Sync_Used": sync_status
        })

    #自动记录csv
    df = pd.DataFrame(result)
    save_path = "reports/speed_table.csv"
    os.makedirs("reports", exist_ok=True)

    df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))

    print(f"\n[Benchmark Report] 已记录到 {save_path}")

if __name__ == "__main__":
    run_benchmark()


