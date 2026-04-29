from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
from PIL import Image
import io
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

app = FastAPI()

# -------------------------------
# Load CLIP model once
# -------------------------------
clip_model = SentenceTransformer("clip-ViT-B-32")
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model.to(device)

# -------------------------------
# Utility: download image from URL
# -------------------------------
def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img

# -------------------------------
# Resize candidate to reference size
# -------------------------------
def resize_to_reference(ref_img_np, img_np):
    return cv2.resize(img_np, (ref_img_np.shape[1], ref_img_np.shape[0]))

# -------------------------------
# A. Similarity score using CLIP
# -------------------------------
def compute_similarity(ref_img_pil, img_pil):
    ref_emb = clip_model.encode(ref_img_pil, convert_to_tensor=True).to(device)
    img_emb = clip_model.encode(img_pil, convert_to_tensor=True).to(device)
    cos_sim = F.cosine_similarity(ref_emb, img_emb, dim=0).item()
    return cos_sim

# -------------------------------
# B. Color accuracy
# -------------------------------
def compute_color_accuracy(ref_img_pil, img_pil):
    ref_np = np.array(ref_img_pil)
    img_np = np.array(img_pil)
    img_np = resize_to_reference(ref_np, img_np)

    ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    diff = np.mean(np.linalg.norm(ref_lab - img_lab, axis=2))
    score = max(0, 1 - diff / 100)
    return float(score)

# -------------------------------
# C. Stroke alignment
# -------------------------------
def compute_stroke_alignment(ref_img_pil, img_pil):
    ref_np = np.array(ref_img_pil)
    img_np = np.array(img_pil)
    img_np = resize_to_reference(ref_np, img_np)

    ref_edge = cv2.Canny(ref_np, 100, 200)
    img_edge = cv2.Canny(img_np, 100, 200)

    return float(ssim(ref_edge, img_edge))

# -------------------------------
# D. Composition score
# -------------------------------
def compute_composition_score(ref_img_pil, img_pil):
    ref_np = np.array(ref_img_pil)
    img_np = np.array(img_pil)
    img_np = resize_to_reference(ref_np, img_np)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Saliency
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(img_np)
    saliency_map = (saliency_map * 255).astype("uint8")

    y, x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    h, w = gray.shape

    # Rule-of-thirds score
    thirds_points = [(w/3, h/3), (2*w/3, h/3), (w/3, 2*h/3), (2*w/3, 2*h/3)]
    dist = min(np.linalg.norm(np.array([x, y]) - np.array(p)) for p in thirds_points)
    thirds_score = max(0, 1 - dist / max(w, h))

    # Symmetry score
    left = gray[:, :w//2].mean()
    right = gray[:, w//2:].mean()
    symmetry_score = 1 - abs(left - right) / 255

    return float((thirds_score + symmetry_score) / 2)

def validate_image_url(url: str) -> bool:
    # Quick check: extension
    if not url.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff')):
        return False
    # Verify HTTP header
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if not response.headers.get("Content-Type", "").startswith("image/"):
            return False
    except Exception:
        return False
    return True

# -------------------------------
# Request body
# -------------------------------
class ImageScoreRequest(BaseModel):
    reference_url: str
    image_urls: List[str]

# -------------------------------
# API endpoint
# -------------------------------
@app.post("/score-images")
def score_images(payload: ImageScoreRequest):
    ref_img = load_image_from_url(payload.reference_url)
    results = []

    for url in payload.image_urls:
        if not validate_image_url(url):
            results.append({
                "url": url,
                "error": "Invalid or non-image URL"
            })
            continue

        img = load_image_from_url(url)

        results.append({
            "url": url,
            "similarity": compute_similarity(ref_img, img),
            "color_accuracy": compute_color_accuracy(ref_img, img),
            "stroke_alignment": compute_stroke_alignment(ref_img, img),
            "composition_score": compute_composition_score(ref_img, img)
        })

    return {"results": results}
