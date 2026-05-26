from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
from PIL import Image
import io
import numpy as np
import os

app = FastAPI()

# Feature flag: run in lightweight mode when heavy deps are not available
LIGHT_MODE = os.environ.get("LIGHT_MODE", "0") in ("1", "true", "True")

# Lazy/lossy placeholders; will be replaced if heavy libs are present
cv2 = None
ssim = None
torch = None
F = None
SentenceTransformer = None
clip_model = None
device = "cpu"

# Try to import heavy deps lazily. If they aren't installed, keep LIGHT_MODE behavior.
def try_import_heavy():
    global cv2, ssim, torch, F, SentenceTransformer, clip_model, device
    try:
        import cv2 as _cv2
        from skimage.metrics import structural_similarity as _ssim
        import torch as _torch
        import torch.nn.functional as _F
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        cv2 = _cv2
        ssim = _ssim
        torch = _torch
        F = _F
        SentenceTransformer = _SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # load the model lazily into clip_model when first needed
        # but if PRELOAD_MODEL set, load now
        if os.environ.get("PRELOAD_MODEL", "0") in ("1", "true", "True"):
            clip_model = SentenceTransformer("clip-ViT-B-32")
            clip_model.to(device)

        return True
    except Exception:
        # heavy imports unavailable; remain in LIGHT_MODE
        return False


# Attempt import now; if it fails we remain in light mode and avoid import-time crashes
_heavy_ok = try_import_heavy()
if not _heavy_ok:
    LIGHT_MODE = True

# -------------------------------
# Utility: download image from URL
# -------------------------------
def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageScoring/1.0)"}
    # Stream and then load to avoid servers that block simple HEAD requests
    response = requests.get(url, stream=True, timeout=15, headers=headers)
    response.raise_for_status()
    # read full content into memory (images are typically small). If needed, we can limit bytes.
    content = response.content
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return img

# -------------------------------
# Resize candidate to reference size
# -------------------------------
def resize_to_reference(ref_img_np, img_np):
    if cv2 is not None:
        return cv2.resize(img_np, (ref_img_np.shape[1], ref_img_np.shape[0]))
    # PIL-based resize fallback
    from PIL import Image as PILImage

    img = PILImage.fromarray(img_np)
    img = img.resize((ref_img_np.shape[1], ref_img_np.shape[0]))
    return np.array(img)

# -------------------------------
# A. Similarity score using CLIP
# -------------------------------
def compute_similarity(ref_img_pil, img_pil):
    """Compute similarity. Uses SentenceTransformer if available. Falls back to a simple histogram comparison in LIGHT_MODE."""
    global clip_model
    if not LIGHT_MODE and SentenceTransformer is not None:
        if clip_model is None:
            clip_model = SentenceTransformer("clip-ViT-B-32")
            clip_model.to(device)
        ref_emb = clip_model.encode(ref_img_pil, convert_to_tensor=True).to(device)
        img_emb = clip_model.encode(img_pil, convert_to_tensor=True).to(device)
        sim = float(F.cosine_similarity(ref_emb, img_emb, dim=0).item())
        return sim

    # Lightweight fallback: color histogram correlation (0..1)
    ref = np.array(ref_img_pil).astype("float32") / 255.0
    img = np.array(img_pil).astype("float32") / 255.0
    ref_hist, _ = np.histogram(ref.flatten(), bins=64, range=(0, 1), density=True)
    img_hist, _ = np.histogram(img.flatten(), bins=64, range=(0, 1), density=True)
    # correlation
    ref_hist = ref_hist / (np.linalg.norm(ref_hist) + 1e-8)
    img_hist = img_hist / (np.linalg.norm(img_hist) + 1e-8)
    return float(np.dot(ref_hist, img_hist))

# -------------------------------
# B. Color accuracy
# -------------------------------
def compute_color_accuracy(ref_img_pil, img_pil):
    ref_np = np.array(ref_img_pil)
    img_np = np.array(img_pil)
    img_np = resize_to_reference(ref_np, img_np)

    if cv2 is not None:
        ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        diff = np.mean(np.linalg.norm(ref_lab - img_lab, axis=2))
        score = max(0, 1 - diff / 100)
        return float(score)

    # Fallback: compare mean colors
    ref_mean = ref_np.mean(axis=(0, 1))
    img_mean = img_np.mean(axis=(0, 1))
    diff = np.linalg.norm(ref_mean - img_mean)
    score = max(0, 1 - diff / 255)
    return float(score)

# -------------------------------
# C. Stroke alignment
# -------------------------------
def compute_stroke_alignment(ref_img_pil, img_pil):
    ref_np = np.array(ref_img_pil)
    img_np = np.array(img_pil)
    img_np = resize_to_reference(ref_np, img_np)

    if cv2 is not None and ssim is not None:
        ref_edge = cv2.Canny(ref_np, 100, 200)
        img_edge = cv2.Canny(img_np, 100, 200)
        return float(ssim(ref_edge, img_edge))

    # Fallback: compare simple gradients
    ref_gray = np.mean(ref_np, axis=2)
    img_gray = np.mean(img_np, axis=2)
    ref_grad = np.abs(np.diff(ref_gray, axis=0)).mean() + np.abs(np.diff(ref_gray, axis=1)).mean()
    img_grad = np.abs(np.diff(img_gray, axis=0)).mean() + np.abs(np.diff(img_gray, axis=1)).mean()
    # similarity from gradients
    return float(1 - abs(ref_grad - img_grad) / (max(ref_grad, img_grad) + 1e-8))

# -------------------------------
# D. Composition score
# -------------------------------
def compute_composition_score(ref_img_pil, img_pil):
    ref_np = np.array(ref_img_pil)
    img_np = np.array(img_pil)
    img_np = resize_to_reference(ref_np, img_np)

    if cv2 is not None:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Saliency
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            _, saliency_map = saliency.computeSaliency(img_np)
            saliency_map = (saliency_map * 255).astype("uint8")
            y, x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
        except Exception:
            # fallback to center
            h, w = img_np.shape[:2]
            y, x = h // 2, w // 2
        h, w = img_np.shape[:2]

        # Rule-of-thirds score
        thirds_points = [(w/3, h/3), (2*w/3, h/3), (w/3, 2*h/3), (2*w/3, 2*h/3)]
        dist = min(np.linalg.norm(np.array([x, y]) - np.array(p)) for p in thirds_points)
        thirds_score = max(0, 1 - dist / max(w, h))

        # Symmetry score
        left = gray[:, :w//2].mean()
        right = gray[:, w//2:].mean()
        symmetry_score = 1 - abs(left - right) / 255

        return float((thirds_score + symmetry_score) / 2)

    # Fallback simple heuristic
    h, w = img_np.shape[:2]
    cx, cy = w / 2, h / 2
    # center mass
    ys, xs = np.indices((h, w))
    mass_x = (xs * img_np.mean(axis=2)).sum() / (img_np.mean(axis=2).sum() + 1e-8)
    mass_y = (ys * img_np.mean(axis=2)).sum() / (img_np.mean(axis=2).sum() + 1e-8)
    dist = np.linalg.norm(np.array([mass_x - cx, mass_y - cy]))
    centered_score = max(0, 1 - dist / max(w, h))
    return float(centered_score)

def validate_image_url(url: str) -> bool:
    # Quick check: common extensions are a fast accept; but don't fail only on extension.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageScoring/1.0)"}
    # If extension looks wrong, we'll still attempt to fetch a small part
    try:
        resp = requests.head(url, allow_redirects=True, timeout=5, headers=headers)
        ct = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and ct.startswith("image/"):
            return True
        # Some servers disallow HEAD or omit content-type; try a small GET
        if resp.status_code in (403, 405) or not ct.startswith("image/"):
            resp = requests.get(url, stream=True, timeout=10, headers=headers)
            resp.raise_for_status()
            # read small initial bytes to validate image signature
            chunk = resp.raw.read(8192) or resp.content[:8192]
            try:
                Image.open(io.BytesIO(chunk)).verify()
                return True
            except Exception:
                return False
    except Exception:
        # Final attempt: direct GET and try to open
        try:
            resp = requests.get(url, stream=True, timeout=10, headers=headers)
            resp.raise_for_status()
            chunk = resp.raw.read(8192) or resp.content[:8192]
            Image.open(io.BytesIO(chunk)).verify()
            return True
        except Exception:
            return False
    return False

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