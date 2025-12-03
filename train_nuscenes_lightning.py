import os
import datetime
import math
import pytorch_lightning as pl
import multiprocessing
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from nuscenes.nuscenes import NuScenes
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import csv

# Import SimpleBEV components
from nets.segnet import Segnet
import utils.vox
import utils.geom

# Import VGGT components for dataset and loss
import sys
sys.path.append('/home/zc/vggt')
from datasets.nuScenesDataset import nuScenesDatasetBEV
from bev.loss import SegmentationLoss, compute_layer_weights
from bev.metrics import IntersectionOverUnion
from bev.seghead import BevEncode
from bev.logger import ImageLogger
from utils.training_log_analysis import parse_csv_and_plot


def seed_everything(seed=42):
    """Fix all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    print(f"[Seed] All random seeds set to {seed} for reproducibility.")


class SimpleBEVSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # Config
        self.opt_cfg = config.model.params.trainer_config
        self.data_cfg = config.model.params.data_config
        self.use_scheduler = True
        
        # Semantic layers
        self.seg_layers = self.data_cfg.semantic_seg.hdmap.layers
        self.num_layers = len(self.seg_layers)
        
        # BEV parameters
        self.XMIN, self.XMAX = self.data_cfg.lift.x_bound[0], self.data_cfg.lift.x_bound[1]
        self.ZMIN, self.ZMAX = self.data_cfg.lift.y_bound[0], self.data_cfg.lift.y_bound[1]
        self.YMIN, self.YMAX = self.data_cfg.lift.z_bound[0], self.data_cfg.lift.z_bound[1]
        self.bounds = (self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX)
        
        self.Z = int((self.XMAX - self.XMIN) / self.data_cfg.lift.x_bound[2])
        self.Y = int((self.YMAX - self.YMIN) / self.data_cfg.lift.z_bound[2])
        self.X = int((self.ZMAX - self.ZMIN) / self.data_cfg.lift.y_bound[2])
        
        # Scene centroid
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 0.0
        scene_centroid_py = np.array([scene_centroid_x, scene_centroid_y, scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid_py).float()
        
        # Vox util
        self.vox_util = utils.vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False
        )
        
        # SimpleBEV model (backbone + view transformation)
        self.segnet = Segnet(
            self.Z, self.Y, self.X, 
            vox_util=self.vox_util,
            use_radar=False,
            use_lidar=False,
            use_metaradar=False,
            do_rgbcompress=True,
            encoder_type='res101',
            rand_flip=False
        )
        
        # Replace the decoder's segmentation head with BevEncode for multi-class segmentation
        # Remove SimpleBEV's original segmentation head
        self.segnet.decoder.segmentation_head = nn.Identity()
        
        # Add BevEncode as the segmentation head (similar to VGGT)
        # BevEncode takes the raw_feat from decoder and outputs num_layers classes
        self.seg_head = BevEncode(inC=128, outC=self.num_layers)
        
        # Loss and metrics
        self.seg_loss = SegmentationLoss()
        self.seg_metric = IntersectionOverUnion(self.num_layers)
        
        # Dataset
        data_split = self.data_cfg['data_split_train']
        if 'mini' in data_split:
            self.nusc = NuScenes(version='v1.0-mini', dataroot=self.data_cfg.data_root, verbose=False)
        else:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_cfg.data_root, verbose=False)
        
        self.log_dir = None
        
    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs):
        """
        Forward pass using SimpleBEV's backbone and view transformation
        """
        # SimpleBEV forward (backbone + view transformation + decoder)
        raw_e, feat_e, seg_e, center_e, offset_e = self.segnet(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=self.vox_util,
            rad_occ_mem0=None
        )
        
        # Use raw_feat from decoder as input to our segmentation head
        # raw_e is the BEV feature from decoder before segmentation heads
        seg_logits = self.seg_head(raw_e)  # [B, num_layers, H, W]
        
        return seg_logits
    
    def prepare_data_from_batch(self, batch):
        """
        Prepare data from batch in SimpleBEV format
        """
        images = batch['image']              # [B, T, S, C, H, W]
        intrinsics = batch['intrinsics']     # [B, T, S, 3, 3]
        extrinsics = batch['extrinsics']     # [B, T, S, 4, 4]
        bev_seg_gt = batch['bev_map_gt']     # [B, num_layers, H, W]
        
        B, T, S, C, H, W = images.shape
        assert T == 1, "Only single frame supported"
        
        # Remove time dimension
        images = images[:, 0]  # [B, S, C, H, W]
        intrinsics = intrinsics[:, 0]  # [B, S, 3, 3]
        extrinsics = extrinsics[:, 0]  # [B, S, 4, 4]
        
        # Images are already normalized in dataset [0, 1], convert to SimpleBEV format [-0.5, 0.5]
        images = images.float()
        if images.max() > 1.5:  # If still in [0, 255]
            images = images / 255.0
        images = images - 0.5
        
        # extrinsics is cam->ego (sensor_to_lidar in dataset)
        # SimpleBEV expects cam0_T_camXs (transformation from each camera to cam0)
        velo_T_cams = extrinsics  # cam->ego
        
        # Get cam0_T_camXs using SimpleBEV's util
        cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
        
        return images, intrinsics, cam0_T_camXs, bev_seg_gt
    
    def shared_step(self, batch, prefix='train'):
        """Shared step for training and validation"""
        images, intrinsics, cam0_T_camXs, bev_seg_gt = self.prepare_data_from_batch(batch)
        
        # Move to device
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        cam0_T_camXs = cam0_T_camXs.to(self.device)
        bev_seg_gt = bev_seg_gt.to(self.device)
        
        # Forward pass
        seg_logits = self.forward(images, intrinsics, cam0_T_camXs)
        
        # Compute loss with layer weights
        weight = compute_layer_weights(bev_seg_gt)
        loss = self.seg_loss(seg_logits, bev_seg_gt, weight, mask_valid=None)
        
        loss_dict = {f'{prefix}/loss_seg': loss.detach()}
        
        # Compute IoU for validation
        if prefix == 'val':
            pred_mask = torch.where(seg_logits > 0, 1, 0)
            gt_mask = (bev_seg_gt + 1) / 2
            self.seg_metric.update(pred_mask, gt_mask)
        
        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, prefix='train')
        
        self.log_dict(loss_dict, prog_bar=True, logger=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        
        metrics = self.trainer.callback_metrics
        epoch = int(self.current_epoch)
        
        fields = [
            "train/loss_seg_epoch",
            "val/loss_seg",
            "val/IoU",
        ]
        
        row = {"epoch": epoch}
        for key in fields:
            val = metrics.get(key)
            row[key] = val.item() if val is not None else None
        
        # Write to CSV
        csv_file = os.path.join(self.log_dir, "train_log.csv")
        # Ensure log directory exists
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["epoch"] + fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: ("" if v is None else f"{v:.5f}") for k, v in row.items()})
        
        # Plot learning curve
        learning_curve = os.path.join(self.log_dir, "loss_plot.png")
        if os.path.exists(csv_file):
            parse_csv_and_plot(csv_file, learning_curve)
    
    def validation_step(self, batch, batch_idx):
        val_epoch = self.opt_cfg.get('val_after_epoch', 0)
        
        if self.current_epoch < val_epoch:
            self.log("val/IoU", torch.tensor(float("nan"), device=self.device),
                     prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
            return
        
        loss, loss_dict = self.shared_step(batch, prefix='val')
        self.log_dict(loss_dict, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
    
    def on_validation_start(self):
        """Clear cache before validation"""
        torch.cuda.empty_cache()
        
    def on_validation_epoch_end(self):
        val_epoch = self.opt_cfg.get('val_after_epoch', 0)
        
        if self.current_epoch < val_epoch:
            return
        
        score = self.seg_metric.compute()
        iou = score.mean().item()
        log_dict = {'val/IoU': iou}
        self.log_dict(log_dict, prog_bar=True, logger=False, on_epoch=True, sync_dist=True)
        
        # Reset metric
        self.seg_metric.reset()
        
        # Clear cache after validation
        torch.cuda.empty_cache()
    
    def log_images(self, batch, split='val', **kwargs):
        """
        Generate visualization images for ImageLogger callback.
        Returns a dict with keys like 'samples', 'reconstructions', etc.
        Each value should be a tensor of shape [B, C, H, W] for visualization.
        """
        log_dict = {}
        
        # Use prepare_data_from_batch to get correct data format
        images, intrinsics, cam0_T_camXs, bev_seg_gt = self.prepare_data_from_batch(batch)
        
        # Move to device
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        cam0_T_camXs = cam0_T_camXs.to(self.device)
        bev_seg_gt = bev_seg_gt.to(self.device)
        
        # Forward pass to get predictions
        with torch.no_grad():
            seg_logits = self.forward(images, intrinsics, cam0_T_camXs)  # [B, C_seg, H, W]
            seg_preds = torch.sigmoid(seg_logits) > 0.5  # Binary predictions
        
        # Convert to float for visualization
        seg_preds = seg_preds.float()      # [B, C_seg, H, W]
        bev_seg_gt = bev_seg_gt.float()    # [B, C_seg, H, W]
        
        # Log ground truth and predictions
        log_dict['ground_truth'] = bev_seg_gt
        log_dict['samples'] = seg_preds
        
        return log_dict
    
    def configure_optimizers(self):
        base_lr = self.opt_cfg.get("lr", 1e-3)
        min_lr = self.opt_cfg.get("min_lr", 5e-4)
        weight_decay = self.opt_cfg.get("weight_decay", 1e-4)
        
        # Optimize both SimpleBEV backbone and segmentation head
        # If you want to freeze the backbone, only optimize seg_head.parameters()
        params = list(self.segnet.parameters()) + list(self.seg_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=weight_decay
        )
        
        # Get total steps
        train_loader = self.train_dataloader()
        steps_per_epoch = len(train_loader)
        num_epochs = self.trainer.max_epochs
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(0.1 * total_steps)
        
        # Scheduler (linear warmup + cosine decay)
        def lr_lambda(current_step):
            if current_step <= warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_ratio = min_lr / base_lr
                return max(cosine_decay, min_ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        print(f"Training steps: {total_steps}, Warmup: {warmup_steps}, Steps per epoch: {steps_per_epoch}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine"
            }
        }
    
    def train_dataloader(self):
        dataset = nuScenesDatasetBEV(
            self.nusc, 'train', self.data_cfg,
            data_split=self.data_cfg['data_split_train']
        )
        
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        return DataLoader(
            dataset,
            batch_size=self.data_cfg['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.data_cfg['num_workers'],
            persistent_workers=True if self.data_cfg['num_workers'] > 0 else False
        )
    
    def val_dataloader(self):
        dataset = nuScenesDatasetBEV(
            self.nusc, 'val', self.data_cfg,
            data_split=self.data_cfg['data_split_val']
        )
        
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False
            )
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=self.data_cfg['batch_size'],
            shuffle=False,
            sampler=sampler,
            num_workers=self.data_cfg['num_workers'],
            persistent_workers=False
        )


def main():
    # Config path
    config_path = './configs/simplebev_seg.yaml'
    
    multiprocessing.set_start_method('spawn', force=True)
    cfg = OmegaConf.load(config_path)
    
    # Set seed
    seed = cfg.model.params.trainer_config.get('seed', 42)
    seed_everything(seed=seed)
    
    # Create model
    model = SimpleBEVSegmentation(cfg)
    
    # Resume from checkpoint if specified
    resume_path = cfg.model.params.trainer_config.resume_path
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    
    # Setup logging
    now = datetime.datetime.now()
    log_folder_path = 'logs/' + '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute)))
    os.makedirs(log_folder_path, exist_ok=True)
    model.log_dir = log_folder_path
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_folder_path,
        monitor='val/IoU',
        filename='best-ckpt-{epoch}-{step}',
        mode='max'
    )
    
    # Image logger callback
    trainer_cfg = cfg.model.params.trainer_config
    logger_freq = trainer_cfg.get('logger_freq', 100)
    image_logger = ImageLogger(
        batch_frequency=logger_freq,
        max_images=4,
        rescale=False,
        log_folder=log_folder_path
    )
    
    # Trainer
    trainer = pl.Trainer(
        strategy="auto",
        accelerator="gpu",
        devices=trainer_cfg.gpu_num,
        precision=trainer_cfg.precision,
        callbacks=[checkpoint_callback, image_logger],
        logger=False,
        max_epochs=trainer_cfg.epochs
    )
    
    # Train!
    trainer.fit(model)


if __name__ == '__main__':
    main()
