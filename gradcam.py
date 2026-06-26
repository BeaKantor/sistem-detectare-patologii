

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import get_val_transforms
from model import load_model

MODEL_PATH = r"C:\xray-project\xray_model.pth"
OUTPUT_DIR = r"C:\xray-project\results\gradcam"

DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]

SEV_LABELS = ['N/A', 'mild', 'moderate', 'severe']


DETECTION_THRESHOLD = 0.5


class GradCAM:
    

    def __init__(self, model: torch.nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.features.denseblock4

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, disease_idx: int) -> np.ndarray:
        
        self.model.zero_grad()

       
        prob_out, _ = self.model(image_tensor)

        
        score = prob_out[0, disease_idx]
        score.backward()

        
        weights = self.gradients.mean(dim=[2, 3])[0]

        
        cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)

       
        cam = F.relu(cam)

        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam


def load_image(image_path: str) -> tuple:
   
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform  = get_val_transforms()
    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0)

    return img_rgb, img_tensor


def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    
    img_resized = cv2.resize(img_rgb, (224, 224))

    
    heatmap_uint8  = np.uint8(255 * heatmap)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb    = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

   
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay


def save_heatmap_figure(
    img_rgb:   np.ndarray,
    overlay:   np.ndarray,
    disease:   str,
    prob:      float,
    severity:  str,
    out_path:  str,
):
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.resize(img_rgb, (224, 224)))
    axes[0].set_title('Original X-Ray', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(
        f'{disease.title()}\nProbability: {prob*100:.1f}% | Severity: {severity}',
        fontsize=11
    )
    axes[1].axis('off')

    plt.suptitle('Grad-CAM Visualization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_gradcam(image_path: str, device: torch.device):
    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir    = os.path.join(OUTPUT_DIR, image_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing: {image_path}")

    
    img_rgb, img_tensor = load_image(image_path)
    img_tensor = img_tensor.to(device)

    
    model  = load_model(MODEL_PATH, device)
    gradcam = GradCAM(model)

    
    model.eval()
    with torch.no_grad():
        prob_out, sev_out = model(img_tensor)

    probs     = prob_out[0].cpu().numpy()
    severities = sev_out[0].argmax(dim=1).cpu().numpy()

    print(f"\nPredictions:")
    print(f"  {'Disease':<30} {'Prob':>8}  {'Severity':>10}  {'Detected':>10}")
   

    detected = []
    for i, disease in enumerate(DISEASES):
        prob     = probs[i]
        sev_label = SEV_LABELS[severities[i]]
        detected_flag = prob >= DETECTION_THRESHOLD
        print(f"  {disease:<30} {prob:>7.1%}  {sev_label:>10}  {'yes' if detected_flag else 'no':>10}")
        if detected_flag:
            detected.append((i, disease, prob, sev_label))

    if not detected:
        print("\nNo diseases detected above threshold. Generating heatmap for highest probability disease.")
        top_idx  = int(np.argmax(probs))
        detected = [(top_idx, DISEASES[top_idx], probs[top_idx], SEV_LABELS[severities[top_idx]])]

    print(f"\nGenerating Grad-CAM heatmaps for {len(detected)} detected disease(s)...")

    for disease_idx, disease, prob, sev_label in detected:
        
        img_tensor_grad = img_tensor.clone().requires_grad_(True)
        heatmap = gradcam.generate(img_tensor_grad, disease_idx)
        overlay = overlay_heatmap(img_rgb, heatmap)

        out_path = os.path.join(out_dir, f"{disease.replace(' ', '_')}_heatmap.png")
        save_heatmap_figure(img_rgb, overlay, disease, prob, sev_label, out_path)
        print(f"  Saved → {out_path}")

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Grad-CAM heatmap generator for chest X-ray disease detection")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray PNG image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_gradcam(args.image, device)