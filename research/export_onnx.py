"""Export utility to compile a trained PyTorch walkability checkpoint to ONNX format."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from research.train_v2 import PathVisionSegModel

def export():
    print("Initializing ONNX Export for PathVision v2.0...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture
    model = PathVisionSegModel()
    ckpt_path = PROJECT_ROOT / "research" / "training_v2" / "checkpoints" / "best_model_v2.pth"
    
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        print(f"Loaded weights from: {ckpt_path}")
    else:
        print("Warning: best_model_v2.pth not found. Exporting base/uninitialized architecture to ONNX.")
        
    model = model.to(device)
    model.eval()
    
    out_dir = PROJECT_ROOT / "research" / "training_v2" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "pathvision_v2.onnx"
    
    dummy_input = torch.randn(1, 3, 240, 320, device=device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"ONNX model successfully compiled and saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX compilation failed: {e}")

if __name__ == "__main__":
    export()
