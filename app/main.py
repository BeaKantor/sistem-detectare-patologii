

import os
import sys
import cv2
import base64
import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import get_val_transforms
from model import load_model
from severity_model import load_severity_model

PROJECT_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH         = os.path.join(PROJECT_ROOT, "xray_model.pth")
SEVERITY_MODEL_PATH = os.path.join(PROJECT_ROOT, "severity_model.pth")


DISEASES = [
    'enlarged cardiomediastinum', 'cardiomegaly', 'atelectasis',
    'consolidation', 'lung edema', 'fracture', 'lung lesion',
    'pleural effusion', 'pneumonia', 'pneumothorax',
    'support device', 'lung opacity', 'pleural other'
]


EXCLUDED_DISEASES = {'support device'}


SEV_LABELS = ['N/A', 'mild', 'moderate', 'severe']

DETECTION_THRESHOLD = 0.5


app = FastAPI(
    title="Chest X-Ray Disease Detection",
    description="Detects 13 chest diseases and their severity from X-ray images.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model           = None
severity_model  = None
gradcam         = None



class GradCAM:
    
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target = model.features.denseblock4
        target.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, image_tensor, disease_idx):
        self.model.zero_grad()
        prob_out, _ = self.model(image_tensor)
        prob_out[0, disease_idx].backward()
        weights = self.gradients.mean(dim=[2, 3])[0]
        cam     = (weights[:, None, None] * self.activations[0]).sum(dim=0)
        cam     = F.relu(cam)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam.cpu().numpy(), (224, 224))
        return cam


@app.on_event("startup")
async def startup_event():
    global model, severity_model, gradcam
    model = load_model(MODEL_PATH, device)
    print(f"Main model loaded on {device}")

    if os.path.exists(SEVERITY_MODEL_PATH):
        severity_model = load_severity_model(SEVERITY_MODEL_PATH, device)
        print(f"Severity model loaded on {device}")
    else:
        severity_model = None
        print(f"WARNING: severity model not found at {SEVERITY_MODEL_PATH}")

    gradcam = GradCAM(model)



def preprocess_image(image_bytes: bytes):
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image.")
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform  = get_val_transforms()
    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    return img_rgb, img_tensor


def predict_severity(img_tensor, disease_idx: int) -> str:
    if severity_model is None:
        return "nedeterminată"

    idx_tensor = torch.tensor([disease_idx], dtype=torch.long, device=device)
    proba      = severity_model.predict_proba(img_tensor, idx_tensor)  # (1, 4)
    sev_class  = int(proba[0].argmax().item())
    return SEV_LABELS[sev_class]


def generate_heatmap_b64(img_rgb: np.ndarray, heatmap: np.ndarray) -> str:
    img_resized   = cv2.resize(img_rgb, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay       = cv2.addWeighted(img_resized, 0.6, heatmap_rgb, 0.4, 0)
    overlay_bgr   = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer     = cv2.imencode('.png', overlay_bgr)
    return base64.b64encode(buffer).decode('utf-8')



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        img_rgb, img_tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


    with torch.no_grad():
        prob_out, _ = model(img_tensor)

    probs = prob_out[0].cpu().numpy()

    detected     = []
    all_diseases = []

    for i, disease in enumerate(DISEASES):
        if disease in EXCLUDED_DISEASES:
            continue

        prob        = float(probs[i])
        is_detected = prob >= DETECTION_THRESHOLD

        all_diseases.append({
            "disease":     disease,
            "probability": round(prob * 100, 1),
            "detected":    is_detected,
        })

        if is_detected:
            severity = predict_severity(img_tensor, i)

        
            img_tensor_grad = img_tensor.clone().requires_grad_(True)
            heatmap     = gradcam.generate(img_tensor_grad, i)
            heatmap_b64 = generate_heatmap_b64(img_rgb, heatmap)

            detected.append({
                "disease":     disease,
                "probability": round(prob * 100, 1),
                "severity":    severity,
                "heatmap":     heatmap_b64,
            })

    detected.sort(key=lambda x: x["probability"], reverse=True)

    return JSONResponse({
        "detected":     detected,
        "all_diseases": all_diseases,
        "device":       str(device),
        "n_detected":   len(detected),
    })


@app.get("/health")
async def health():
    return {
        "status":               "ok",
        "device":               str(device),
        "main_model_loaded":    model is not None,
        "severity_model_loaded": severity_model is not None,
    }


@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))