"""PathVision Version 2 - Transfer Learning & Fine-tuning Pipeline template.

This script implements the PyTorch training pipeline for transfer learning on the
SUNRGBD dataset. It loads the base segmentation weights, freezes the encoder,
computes losses on generated walkable masks, and handles mixed precision, early stopping,
and ONNX compilation.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            pass
        def close(self):
            pass

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Class mapping (0 = Unsafe/Obstacle, 1 = Safe Walkable Path, 2 = Path Boundary)
NUM_CLASSES = 3

class SUNRGBDSegDataset(Dataset):
    """PyTorch Dataset class for loading SUNRGBD samples and generating floor masks."""
    
    def __init__(self, dataset_dir: Path, folders: list[Path], transform=None):
        self.dataset_dir = dataset_dir
        self.folders = folders
        self.transform = transform
        
    def __len__(self):
        return len(self.folders)
        
    def __getitem__(self, idx):
        folder = self.folders[idx]
        img_name = folder.name
        
        # Load RGB image
        img_path = folder / "image" / f"{img_name}.jpg"
        img = scipy.io.loadmat(str(folder / "seg.mat"))  # we use seg.mat labels
        seg_label = img['seglabel']
        
        # In Matlab format: floor = 11, floor mat = 143
        floor_mask = (seg_label == 11) | (seg_label == 143)
        
        # Load image via PIL/OpenCV or PyTorch tensor (simplified mock load)
        # In production, load actual image and resize both to 320x240
        input_tensor = torch.randn(3, 240, 320, dtype=torch.float32)
        
        # Target mask: class 0 = background, class 1 = walkable floor (floor_mask)
        target_mask = torch.from_numpy(floor_mask).long()
        # Resize target to 240x320
        target_mask = nn.functional.interpolate(
            target_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(240, 320),
            mode="nearest"
        ).squeeze().long()
        
        return input_tensor, target_mask

class PathVisionSegModel(nn.Module):
    """Lightweight Encoder-Decoder segmentation model matching v1.0 specifications."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        # Simplified MobileNetV3-style backbone encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # Decoder upsampling layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits

def train_v2(args):
    print("Initializing PathVision Version 2 training pipeline...")
    
    # Establish research output directories
    research_dir = PROJECT_ROOT / "research" / "training_v2"
    ckpt_dir = research_dir / "checkpoints"
    tb_dir = research_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # Load dataset
    dataset_path = Path(args.dataset_path)
    folders = sorted([p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("NYU")])
    
    # Validation split (80% train, 20% validation)
    split_idx = int(len(folders) * 0.8)
    train_folders = folders[:split_idx]
    val_folders = folders[split_idx:]
    
    train_dataset = SUNRGBDSegDataset(dataset_path, train_folders)
    val_dataset = SUNRGBDSegDataset(dataset_path, val_folders)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Instantiate Model
    model = PathVisionSegModel(num_classes=NUM_CLASSES)
    
    # Transfer learning: load base weights if they exist
    base_weights_path = PROJECT_ROOT / "models" / "best_model.pth"
    if base_weights_path.exists():
        try:
            # Load state dict (matching keys)
            state_dict = torch.load(str(base_weights_path), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded base model weights from: {base_weights_path}")
            
            # Freeze encoder layers to preserve pre-trained features (knowledge preservation)
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Frozen encoder backbone for fine-tuning.")
        except Exception as e:
            print(f"Error loading base weights: {e}. Starting training from scratch.")
            
    model = model.to(device)
    
    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    
    # Mixed precision gradient scaler
    scaler = GradScaler()
    
    # Tensorboard logger
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    # Early stopping config
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Resume support
    start_epoch = 0
    resume_path = ckpt_dir / "latest_checkpoint.pth"
    if args.resume and resume_path.exists():
        checkpoint = torch.load(str(resume_path))
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        print(f"Resumed training from epoch {start_epoch}")
        
    # Epoch Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation epoch run
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_loss / len(val_dataset)
        
        scheduler.step(epoch_val_loss)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)
        print(f"Epoch {epoch}/{args.epochs - 1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Save latest checkpoint (for resume support)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_val_loss
        }, str(ckpt_dir / "latest_checkpoint.pth"))
        
        # Save best model and handle early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), str(ckpt_dir / "best_model_v2.pth"))
            patience_counter = 0
            print(f"Saved new best model checkpoint to best_model_v2.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
    writer.close()
    
    # Export fine-tuned model to ONNX
    onnx_path = ckpt_dir / "pathvision_v2.onnx"
    print(f"Exporting model to ONNX format at {onnx_path}...")
    dummy_input = torch.randn(1, 3, 240, 320, device=device)
    model.eval()
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )
        print("ONNX export successfully completed.")
    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathVision V2 Fine-tuning")
    parser.add_argument("--dataset_path", type=str, default=str(DATASET_DIR), help="Path to NYUdata folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    train_v2(args)
