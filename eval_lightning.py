"""
SimpleBEV Segmentation Evaluation Script (超简化版)
直接使用 train_nuscenes_lightning.py 中的类
"""
import pytorch_lightning as pl
import multiprocessing
from omegaconf import OmegaConf
import argparse
import sys
import datetime
import random
import numpy as np
import torch
import time

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
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='./configs/simplebev_seg.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'predict'],
                        help='Mode: evaluate or predict')
    
    args = parser.parse_args()
    
    multiprocessing.set_start_method('spawn', force=True)
    
    if args.mode == 'evaluate':
        evaluate(args.checkpoint, args.config)
    else:
        predict(args.checkpoint, args.config)
