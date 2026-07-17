"""Loss functions for training the walkability safe-path segmenter in PathVision v2.0."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for multiclass semantic segmentation (minimizes global overlap error)."""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits shape: (N, C, H, W), targets shape: (N, H, W)
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_onehot, dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - torch.mean(dice)

class FocalLoss(nn.Module):
    """Focal Loss to focus training on hard boundary/misclassified pixels."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss per pixel
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)

def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1: # cover zero division
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax Loss (directly optimizes Jaccard index / IoU bounds)."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W), targets: (N, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            prob_c = probs[:, c, :, :]
            
            # Flatten to 1D
            prob_flat = prob_c.reshape(-1)
            target_flat = target_c.reshape(-1)
            
            # Compute errors
            errors = torch.abs(target_flat - prob_flat)
            errors_sorted, perm = torch.sort(errors, descending=True)
            target_sorted = target_flat[perm]
            
            # Calculate gradient
            grad = lovasz_grad(target_sorted)
            losses.append(torch.dot(errors_sorted, grad))
            
        return torch.mean(torch.stack(losses))

class CombinedNavigationLoss(nn.Module):
    """Combined Multi-Task Loss for physical walkability and sharp boundaries."""
    
    def __init__(self, w_lovasz: float = 1.0, w_focal: float = 1.0, w_dice: float = 0.5):
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        
        self.w_lovasz = w_lovasz
        self.w_focal = w_focal
        self.w_dice = w_dice
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_lovasz = self.lovasz(logits, targets)
        l_focal = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        
        # Combined loss
        total_loss = (self.w_lovasz * l_lovasz) + (self.w_focal * l_focal) + (self.w_dice * l_dice)
        return total_loss

if __name__ == "__main__":
    # Self-test loss computations
    print("Testing Walkability Loss Functions...")
    dummy_logits = torch.randn(2, 3, 240, 320)  # Batch=2, Classes=3, Size=240x320
    dummy_targets = torch.randint(0, 3, (2, 240, 320))
    
    loss_fn = CombinedNavigationLoss()
    val = loss_fn(dummy_logits, dummy_targets)
    print(f"Combined Loss calculated successfully: {val.item():.4f}")
