"""
SimpleBEV Segmentation Evaluation Script (超简化版)
直接使用 train_nuscenes_lightning.py 中的类
"""
import pytorch_lightning as pl
import multiprocessing
from omegaconf import OmegaConf
import argparse
import csv
import os
import sys
import datetime
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import Any
from torch.utils.data import DataLoader

# 修复 vggt 导入路径问题 - 在导入 train_nuscenes_lightning 之前
sys.path.insert(0, '/home/zc/vggt')
# 将正确的 geometry 函数注入到错误的导入路径中
import vggt.utils.geometry as correct_geometry
if 'vggt.utils' not in sys.modules:
    import vggt.utils
    sys.modules['vggt.utils'] = vggt.utils
# 将正确路径的 geometry 模块设置为导入目标
sys.modules['vggt.utils.geometry'] = correct_geometry

# Import training module - 直接复用所有功能
from train_nuscenes_lightning import SimpleBEVSegmentation
from datasets.nuScenesDataset import nuScenesDatasetBEV


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = './configs/simplebev_seg.yaml'
DEFAULT_CKPT_PATH = './logs/12_28_19_59/best-ckpt-epoch=49-step=171850.ckpt'


def print_model_summary(model):
    """打印模型参数统计"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("-" * 60)
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print(f"Model Size (MB):          {total_params * 4 / (1024**2):.2f}")
    print("="*60 + "\n")


def seed_everything(seed=42):
    """Fix all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    print(f"[Seed] All random seeds set to {seed} for reproducibility.")


def resolve_project_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def make_log_dir(suffix: str | None = None) -> Path:
    name = '_'.join('%02d' % value for value in datetime.datetime.now().timetuple()[1:5])
    if suffix:
        name = f'{name}_{suffix}'
    log_dir = REPO_ROOT / 'logs' / name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Log] Logging to {log_dir.relative_to(REPO_ROOT)}")
    return log_dir


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def infer_batch_size(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            try:
                return infer_batch_size(value)
            except ValueError:
                continue
    if isinstance(batch, (tuple, list)):
        for value in batch:
            try:
                return infer_batch_size(value)
            except ValueError:
                continue
    raise ValueError('Cannot infer batch size from batch.')


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def bytes_to_mb(value: int | None) -> float | None:
    return None if value is None else value / (1024.0 ** 2)


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reset_predict_state(model: torch.nn.Module) -> None:
    metric = getattr(model, 'seg_metric', None)
    if metric is not None and hasattr(metric, 'reset'):
        metric.reset()
    if hasattr(model, 'inference_times'):
        model.inference_times = []


def pad_images_to_multiple(images: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    height, width = images.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return images
    return F.pad(images, (0, pad_width, 0, pad_height), mode='constant', value=0.0)


def speed_forward(model: SimpleBEVSegmentation, batch: dict[str, Any]) -> torch.Tensor:
    device = next(model.parameters()).device
    images, intrinsics, cam0_T_camXs, _ = model.prepare_data_from_batch(batch)
    images = images.to(device, non_blocking=True)
    images = pad_images_to_multiple(images)
    intrinsics = intrinsics.to(device, non_blocking=True)
    cam0_T_camXs = cam0_T_camXs.to(device, non_blocking=True)
    return model.forward(images, intrinsics, cam0_T_camXs)


def profile_flops(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[int | None, int | None, str]:
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001
        return None, None, f'torch.profiler unavailable: {exc}'

    try:
        batch = next(iter(loader))
    except StopIteration:
        return None, None, 'dataloader is empty'

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    batch = move_to_device(batch, device)
    batch_size = infer_batch_size(batch)
    try:
        synchronize(device)
        with torch.inference_mode():
            with profile(activities=activities, record_shapes=True, with_flops=True) as profiler:
                speed_forward(model, batch)
        synchronize(device)
        flops = sum(int(getattr(event, 'flops', 0) or 0) for event in profiler.key_averages())
        reset_predict_state(model)
        return flops, flops // max(batch_size, 1), 'torch.profiler with_flops=True; unsupported ops are not counted'
    except Exception as exc:  # noqa: BLE001
        reset_predict_state(model)
        return None, None, f'FLOPs profiling failed: {exc}'


def measure_speed(model: torch.nn.Module, loader: DataLoader, args: argparse.Namespace,
                  device: torch.device) -> dict[str, Any]:
    model.eval().to(device)
    torch.backends.cudnn.benchmark = True
    reset_predict_state(model)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    samples = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            if batch_index < args.warmup_batches:
                speed_forward(model, batch)
                continue

            if len(times) == 0 and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)

            batch_size = infer_batch_size(batch)
            synchronize(device)
            start = time.perf_counter()
            speed_forward(model, batch)
            synchronize(device)
            times.append(time.perf_counter() - start)
            samples += batch_size

            if len(times) >= args.measure_batches:
                break

    if not times:
        raise RuntimeError('No timed batches were collected. Reduce --warmup-batches or check the dataset.')

    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
    total_params, trainable_params = count_parameters(model)

    if args.skip_flops:
        flops, flops_per_sample, flops_note = None, None, 'skipped by --skip-flops'
    else:
        flops, flops_per_sample, flops_note = profile_flops(model, loader, device)

    reset_predict_state(model)
    total_time = float(sum(times))
    return {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'split': 'val',
        'device': str(device),
        'batch_size': int(args.batch_size),
        'warmup_batches': int(args.warmup_batches),
        'measured_batches': len(times),
        'measured_samples': samples,
        'avg_per_batch_ms': total_time / len(times) * 1000.0,
        'avg_per_sample_ms': total_time / samples * 1000.0,
        'latency_ms': total_time / samples * 1000.0,
        'fps': samples / total_time,
        'param_count': total_params,
        'param_million': total_params / 1e6,
        'trainable_param_count': trainable_params,
        'trainable_param_million': trainable_params / 1e6,
        'peak_memory_allocated_mb': bytes_to_mb(peak_allocated),
        'peak_memory_reserved_mb': bytes_to_mb(peak_reserved),
        'flops': flops,
        'gflops': None if flops is None else flops / 1e9,
        'flops_per_sample': flops_per_sample,
        'gflops_per_sample': None if flops_per_sample is None else flops_per_sample / 1e9,
        'flops_note': flops_note,
    }


def write_speed_metrics_csv(log_dir: Path, metrics: dict[str, Any], paths: dict[str, Path]) -> Path:
    csv_path = log_dir / 'simplebev_speed_metrics.csv'
    row = {**metrics, **{key: str(value) for key, value in paths.items()}}
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({key: '' if value is None else value for key, value in row.items()})
    return csv_path


def print_speed_summary(metrics: dict[str, Any], csv_path: Path) -> None:
    flops_text = 'N/A' if metrics['gflops_per_sample'] is None else f"{metrics['gflops_per_sample']:.2f}"
    memory_text = 'N/A' if metrics['peak_memory_allocated_mb'] is None else f"{metrics['peak_memory_allocated_mb']:.2f}"
    print(
        f"[Speed:SimpleBEV] Params (M): {metrics['param_million']:.2f} | "
        f"FLOPs (G): {flops_text} | Peak Mem. (MB): {memory_text} | "
        f"Latency (ms): {metrics['avg_per_sample_ms']:.2f}"
    )
    print(
        f"[Speed:SimpleBEV] avg per-sample: {metrics['avg_per_sample_ms']:.2f} ms | "
        f"FPS: {metrics['fps']:.2f} | samples={metrics['measured_samples']} | "
        f"batches={metrics['measured_batches']}"
    )
    print(f"[Speed:SimpleBEV] avg per-batch: {metrics['avg_per_batch_ms']:.2f} ms")
    print(
        f"[Params:SimpleBEV] total={metrics['param_million']:.2f}M | "
        f"trainable={metrics['trainable_param_million']:.2f}M"
    )
    if metrics['peak_memory_allocated_mb'] is not None:
        print(
            f"[Memory:SimpleBEV] peak allocated: {metrics['peak_memory_allocated_mb']:.2f} MiB | "
            f"reserved/cache: {metrics['peak_memory_reserved_mb']:.2f} MiB"
        )
    else:
        print('[Memory:SimpleBEV] CUDA is unavailable; GPU memory metrics were not collected.')
    if metrics['gflops_per_sample'] is not None:
        print(
            f"[FLOPs:SimpleBEV] {metrics['gflops_per_sample']:.2f} GFLOPs per sample "
            f"({metrics['gflops']:.2f} GFLOPs per batch)"
        )
    else:
        print(f"[FLOPs:SimpleBEV] unavailable ({metrics['flops_note']})")
    print(f"[Speed:SimpleBEV] metrics csv -> {csv_path.relative_to(REPO_ROOT)}")


def build_speed_dataloader(model: SimpleBEVSegmentation, args: argparse.Namespace) -> DataLoader:
    data_cfg = model.data_cfg
    dataset = nuScenesDatasetBEV(
        model.nusc, 'val', data_cfg,
        data_split=data_cfg['data_split_val']
    )
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg['num_workers'])
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    print(f"[Data] split={data_cfg['data_split_val']} | bs={args.batch_size} | workers={num_workers} | n={len(dataset)}")
    return loader


def measure_speed_entry(checkpoint_path: str, config_path: str, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = resolve_project_path(checkpoint_path)
    config_path = resolve_project_path(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.load(config_path)
    data_cfg = cfg.model.params.data_config
    if args.batch_size is None:
        args.batch_size = int(data_cfg.get('batch_size', 1))
    if args.batch_size != 1:
        print(f"[Speed] --measure-speed forces batch_size=1; ignoring --batch-size={args.batch_size}")
    args.batch_size = 1
    data_cfg.batch_size = 1
    if args.num_workers is not None:
        data_cfg.num_workers = int(args.num_workers)

    seed = cfg.model.params.trainer_config.get('seed', 42)
    seed_everything(seed=seed)

    model = SimpleBEVSegmentation.load_from_checkpoint(str(checkpoint_path), config=cfg, strict=False)
    print_model_summary(model)
    log_dir = make_log_dir('speed')
    loader = build_speed_dataloader(model, args)
    device = resolve_device(args.device)
    metrics = measure_speed(model, loader, args, device)
    csv_path = write_speed_metrics_csv(
        log_dir,
        metrics,
        {'config': config_path, 'ckpt': checkpoint_path},
    )
    print_speed_summary(metrics, csv_path)
    return metrics


def evaluate(checkpoint_path, config_path='./configs/simplebev_seg.yaml'):
    """使用 trainer.validate() 快速评估"""
    cfg = OmegaConf.load(config_path)
    
    model = SimpleBEVSegmentation.load_from_checkpoint(checkpoint_path, config=cfg)
    model.eval()
    
    # 打印模型参数统计
    print_model_summary(model)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision=cfg.model.params.trainer_config.precision
    )
    
    results = trainer.validate(model)
    return results


def predict(checkpoint_path, config_path='./configs/simplebev_seg.yaml'):
    """使用 trainer.predict() 保存预测结果，包含推理速度统计"""
    import os
    import torch
    import json
    import sys
    import time
    sys.path.append('/home/zc/vggt')
    from bev.metrics import IntersectionOverUnion
    from bev.logger import ImageLogger
    
    cfg = OmegaConf.load(config_path)

    # Set seed
    seed = cfg.model.params.trainer_config.get('seed', 42)
    seed_everything(seed=seed)
    
    model = SimpleBEVSegmentation.load_from_checkpoint(checkpoint_path, config=cfg)
    
    # 打印模型参数统计
    print_model_summary(model)
    
    # 初始化推理时间列表
    inference_times = []
    
    # 添加 predict_step (带时间统计)
    def predict_step(self, batch, batch_idx):
        start = time.time()
        
        images, intrinsics, cam0_T_camXs, bev_seg_gt = self.prepare_data_from_batch(batch)
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        cam0_T_camXs = cam0_T_camXs.to(self.device)
        
        with torch.no_grad():
            seg_logits = self.forward(images, intrinsics, cam0_T_camXs)
            seg_probs = torch.sigmoid(seg_logits)
            seg_preds = seg_probs > 0.5
        
        # 同步GPU，确保计时准确
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        inference_times.append(elapsed)
        
        return {
            'predictions': seg_preds.cpu(),
            'probabilities': seg_probs.cpu(),
            'ground_truth': bev_seg_gt.cpu()
        }
    
    # 添加 predict_dataloader (复用 val_dataloader)
    def predict_dataloader(self):
        return self.val_dataloader()
    
    import types
    model.predict_step = types.MethodType(predict_step, model)
    model.predict_dataloader = types.MethodType(predict_dataloader, model)
    model.eval()

    # create logger path
    now = datetime.datetime.now()
    log_folder_path ='logs/' + '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute)))
    os.makedirs(log_folder_path, exist_ok=True)
    print(f"[Log] Logging to {log_folder_path}")

    # Image logger callback
    trainer_cfg = cfg.model.params.trainer_config
    logger_freq = trainer_cfg.get('logger_freq', 100)
    image_logger = ImageLogger(
        batch_frequency=logger_freq,
        max_images=4,
        rescale=False,
        log_folder=log_folder_path
    )
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[image_logger],
        precision=cfg.model.params.trainer_config.precision
    )
    
    # 使用 trainer.predict() - 自动调用 model.predict_dataloader()
    predictions = trainer.predict(model)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(checkpoint_path), 'predictions_val')
    os.makedirs(output_dir, exist_ok=True)
    
    all_preds = torch.cat([p['predictions'] for p in predictions], dim=0)
    all_probs = torch.cat([p['probabilities'] for p in predictions], dim=0)
    all_gts = torch.cat([p['ground_truth'] for p in predictions], dim=0)
    
    # 计算指标
    seg_metric = IntersectionOverUnion(all_preds.shape[1])
    pred_mask = all_preds.float()
    gt_mask = (all_gts + 1) / 2
    seg_metric.update(pred_mask, gt_mask)
    iou_scores = seg_metric.compute()
    mean_iou = iou_scores.mean().item()
    
    # 打印IoU结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    layer_names = cfg.model.params.data_config.semantic_seg.hdmap.layers
    for i, iou in enumerate(iou_scores):
        print(f"IoU {layer_names[i]}: {iou:.4f}")
    print("-"*60)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("="*60)
    
    # 打印推理速度统计
    total_time = sum(inference_times)
    num_samples = len(inference_times)
    avg_time = total_time / num_samples
    fps = 1.0 / avg_time
    batch_size = cfg.model.params.data_config.batch_size
    
    print("\n" + "="*60)
    print("INFERENCE SPEED")
    print("="*60)
    print(f"Total samples:            {num_samples}")
    print(f"Batch size:               {batch_size}")
    print(f"Total inference time:     {total_time:.2f} s")
    print(f"Average time per batch:   {avg_time:.4f} s")
    print(f"FPS (batches/sec):        {fps:.2f}")
    print(f"FPS (samples/sec):        {fps * batch_size:.2f}")
    print("="*60 + "\n")
    
    # 保存metrics到json
    metrics_dict = {
        'mean_iou': mean_iou,
        'per_layer_iou': {layer_names[i]: iou.item() for i, iou in enumerate(iou_scores)},
        'inference': {
            'total_samples': num_samples,
            'batch_size': batch_size,
            'total_time_sec': total_time,
            'avg_time_per_batch_sec': avg_time,
            'fps_batch': fps,
            'fps_sample': fps * batch_size
        }
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"✓ Metrics saved to {output_dir}/metrics.json")
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimpleBEV Evaluation')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CKPT_PATH,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'predict'],
                        help='Mode: evaluate or predict')
    parser.add_argument('--measure-speed', action='store_true',
                        help='Report Params, FLOPs, peak CUDA memory, latency, and FPS.')
    parser.add_argument('--warmup-batches', type=int, default=20, help='Warmup batches excluded from timing.')
    parser.add_argument('--measure-batches', type=int, default=100, help='Timed batches.')
    parser.add_argument('--skip-flops', action='store_true', help='Skip torch.profiler FLOPs profiling.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size; speed mode forces 1.')
    parser.add_argument('--num-workers', type=int, default=None, help='Override dataloader workers.')
    parser.add_argument('--device', default='auto', help='auto, cuda, cuda:0, or cpu.')
    
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn', force=True)

    if args.measure_speed:
        measure_speed_entry(args.checkpoint, args.config, args)
        raise SystemExit(0)
    
    if args.mode == 'evaluate':
        evaluate(args.checkpoint, args.config)
    else:
        predict(args.checkpoint, args.config)
