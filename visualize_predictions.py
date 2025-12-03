"""
Visualization script for SimpleBEV Lightning predictions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/zc/vggt')

from omegaconf import OmegaConf
from train_nuscenes_lightning import SimpleBEVSegmentation
import os


def visualize_prediction(model, batch, save_path='visualization.png'):
    """
    Visualize BEV segmentation predictions
    
    Args:
        model: trained SimpleBEVSegmentation model
        batch: data batch
        save_path: path to save visualization
    """
    model.eval()
    
    with torch.no_grad():
        images, intrinsics, cam0_T_camXs, bev_seg_gt = model.prepare_data_from_batch(batch)
        
        # Move to device
        images = images.cuda()
        intrinsics = intrinsics.cuda()
        cam0_T_camXs = cam0_T_camXs.cuda()
        
        # Forward pass
        seg_logits = model.forward(images, intrinsics, cam0_T_camXs)
        
        # Get predictions
        pred_mask = torch.sigmoid(seg_logits) > 0.5
        gt_mask = (bev_seg_gt + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
    # Move to CPU and convert to numpy
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    images = images.cpu().numpy()
    
    # Get layer names
    layer_names = model.seg_layers
    num_layers = len(layer_names)
    
    # Create visualization
    batch_size = pred_mask.shape[0]
    sample_idx = 0  # Visualize first sample in batch
    
    fig, axes = plt.subplots(3, num_layers, figsize=(num_layers * 4, 12))
    
    # Plot each layer
    for i, layer_name in enumerate(layer_names):
        # Ground truth
        axes[0, i].imshow(gt_mask[sample_idx, i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'GT: {layer_name}')
        axes[0, i].axis('off')
        
        # Prediction
        axes[1, i].imshow(pred_mask[sample_idx, i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Pred: {layer_name}')
        axes[1, i].axis('off')
        
        # Overlay (green=TP, red=FP, blue=FN)
        overlay = np.zeros((gt_mask.shape[2], gt_mask.shape[3], 3))
        tp = (gt_mask[sample_idx, i] > 0.5) & (pred_mask[sample_idx, i] > 0.5)
        fp = (gt_mask[sample_idx, i] <= 0.5) & (pred_mask[sample_idx, i] > 0.5)
        fn = (gt_mask[sample_idx, i] > 0.5) & (pred_mask[sample_idx, i] <= 0.5)
        
        overlay[tp, 1] = 1  # Green for true positive
        overlay[fp, 0] = 1  # Red for false positive
        overlay[fn, 2] = 1  # Blue for false negative
        
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f'Overlay: {layer_name}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")


def compute_iou(pred, gt):
    """Compute IoU for binary masks"""
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def evaluate_model(checkpoint_path, config_path, num_samples=10):
    """
    Evaluate model on validation set and create visualizations
    
    Args:
        checkpoint_path: path to model checkpoint
        config_path: path to config file
        num_samples: number of samples to visualize
    """
    print("Loading model...")
    cfg = OmegaConf.load(config_path)
    model = SimpleBEVSegmentation(cfg)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    
    # Get validation dataloader
    val_loader = model.val_dataloader()
    
    print(f"Evaluating on {len(val_loader.dataset)} samples...")
    
    # Create output directory
    vis_dir = os.path.dirname(checkpoint_path)
    vis_dir = os.path.join(vis_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Evaluate and visualize
    all_ious = {name: [] for name in model.seg_layers}
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_samples:
            break
        
        print(f"Processing batch {batch_idx + 1}/{num_samples}...")
        
        with torch.no_grad():
            images, intrinsics, cam0_T_camXs, bev_seg_gt = model.prepare_data_from_batch(batch)
            
            images = images.cuda()
            intrinsics = intrinsics.cuda()
            cam0_T_camXs = cam0_T_camXs.cuda()
            
            seg_logits = model.forward(images, intrinsics, cam0_T_camXs)
            pred_mask = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
            gt_mask = ((bev_seg_gt + 1) / 2 > 0.5).cpu().numpy()
        
        # Compute IoU for each layer
        for i, layer_name in enumerate(model.seg_layers):
            iou = compute_iou(pred_mask[0, i], gt_mask[0, i])
            all_ious[layer_name].append(iou)
        
        # Create visualization
        save_path = os.path.join(vis_dir, f'sample_{batch_idx:03d}.png')
        visualize_prediction(model, batch, save_path)
    
    # Print IoU statistics
    print("\n" + "=" * 50)
    print("IoU Results:")
    print("=" * 50)
    
    mean_iou_all = []
    for layer_name in model.seg_layers:
        ious = all_ious[layer_name]
        mean_iou = np.mean(ious)
        std_iou = np.std(ious)
        mean_iou_all.append(mean_iou)
        print(f"{layer_name:20s}: {mean_iou:.4f} ± {std_iou:.4f}")
    
    print("-" * 50)
    print(f"{'Mean IoU':20s}: {np.mean(mean_iou_all):.4f}")
    print("=" * 50)
    
    print(f"\nVisualizations saved to {vis_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SimpleBEV predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='./configs/simplebev_seg.yaml',
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.num_samples)
