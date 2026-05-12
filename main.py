from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
from functools import lru_cache
from PIL import Image
import io
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
app = FastAPI()

# -------------------------------
# Lazy-load CLIP model (sentence-transformers)
# This avoids import-time failures when huggingface_hub or
# sentence-transformers versions are incompatible with the
# server environment. The model will be loaded on first use.
# -------------------------------
_clip_model = None
_device = None

def get_clip_model():
    global _clip_model, _device
    if _clip_model is not None:
        return _clip_model, _device

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        # Raise a clearer error when compute_similarity is invoked
        raise RuntimeError(
            "sentence-transformers is not available or incompatible. "
            "Install sentence-transformers and a compatible huggingface_hub (see README)."
        ) from e

    # determine device and load model
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _clip_model = SentenceTransformer("clip-ViT-B-32")
    _clip_model.to(_device)
    return _clip_model, _device


# -------------------------------
# HTTP session with connection pooling and retries
# -------------------------------
_session = None
def get_session():
    global _session
    if _session is not None:
        return _session
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(500, 502, 504))
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    _session = s
    return _session


# ThreadPool for parallel downloads / CPU work
_download_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


# Simple LRU cache for reference embeddings (by URL)
@lru_cache(maxsize=32)
def get_ref_embedding_from_url(reference_url: str):
    # load image then encode; used to avoid recomputing reference embedding
    img = load_image_from_url(reference_url)
    clip_model, device = get_clip_model()
    emb = clip_model.encode(img, convert_to_tensor=True).to(device)
    return emb

# -------------------------------
# Utility: download image from URL
# -------------------------------
def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    s = get_session()
    resp = s.get(url, timeout=timeout)
    resp.raise_for_status()
    # ensure content-type is image
    ctype = resp.headers.get("Content-Type", "")
    if not ctype.startswith("image/"):
        raise ValueError("URL did not return an image content-type")
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def download_images_parallel(urls: List[str], timeout: int = 10):
    """Download images in parallel. Returns list of (url, PIL.Image or Exception)."""
    session = get_session()

    def _fetch(url: str):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            if not resp.headers.get("Content-Type", "").startswith("image/"):
                return url, ValueError("Not an image")
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return url, img
        except Exception as e:
            return url, e

    futures = [
        _download_executor.submit(_fetch, u)
        for u in urls
    ]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
    # as_completed changes order; return results mapped by url order
    url_to_res = {u: r for u, r in results}
    return [ (u, url_to_res.get(u)) for u in urls ]

# -------------------------------
# Resize candidate to reference size
# -------------------------------
def resize_to_reference(ref_img_np, img_np):
    return cv2.resize(img_np, (ref_img_np.shape[1], ref_img_np.shape[0]))

# -------------------------------
# A. Similarity score using CLIP
# -------------------------------
def compute_similarity_batch(ref_img_pil, img_pil_list: List[Image.Image], ref_emb=None):
    """Encode candidates in one batch and return cosine similarities list.
    If ref_emb (a tensor) is provided, reuse it instead of re-encoding the reference.
    """
    clip_model, device = get_clip_model()

    if ref_emb is None:
        imgs = [ref_img_pil] + img_pil_list
        embs = clip_model.encode(imgs, convert_to_tensor=True)
        try:
            embs = embs.to(device)
        except Exception:
            pass
        ref_emb = embs[0]
        cand_embs = embs[1:]
    else:
        # encode only candidates
        cand_embs = clip_model.encode(img_pil_list, convert_to_tensor=True)
        try:
            cand_embs = cand_embs.to(device)
        except Exception:
            pass

    if len(cand_embs) == 0:
        return []

    # ensure tensors have correct shape
    if isinstance(cand_embs, list):
        cand_tensor = torch.stack(cand_embs)
    else:
        cand_tensor = cand_embs

    sims = F.cosine_similarity(ref_emb.unsqueeze(0).repeat(cand_tensor.shape[0], 1), cand_tensor, dim=1)
    return [float(s.item()) for s in sims]

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

    # Saliency: try using OpenCV's saliency (from opencv-contrib). If not available,
    # fall back to a lightweight heuristic using Laplacian magnitude + blur.
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliency_map = saliency.computeSaliency(img_np)
        saliency_map = (saliency_map * 255).astype("uint8")
    except Exception:
        # fallback: use Laplacian + Gaussian blur as a simple saliency proxy
        gray_float = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype("float32")
        lap = np.abs(cv2.Laplacian(gray_float, cv2.CV_32F))
        lap = cv2.GaussianBlur(lap, (9, 9), 0)
        saliency_map = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

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
        s = get_session()
        response = s.head(url, allow_redirects=True, timeout=5)
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
    # Load reference image (fail fast)
    try:
        ref_img = load_image_from_url(payload.reference_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load reference image: {e}")

    # Download all candidate images in parallel
    downloads = download_images_parallel(payload.image_urls)

    # Prepare lists preserving order
    images = []
    results = []
    valid_indices = []

    for i, (url, res) in enumerate(downloads):
        if isinstance(res, Exception):
            results.append({"url": url, "error": str(res)})
        else:
            images.append(res)
            valid_indices.append(i)
            results.append(None)  # placeholder to fill later

    # Compute similarities in batch for successful images, reusing cached ref embedding when possible
    similarities = []
    if images:
        try:
            try:
                ref_emb = get_ref_embedding_from_url(payload.reference_url)
            except Exception:
                # cache miss or model not available yet: compute embedding once and store via function call
                ref_emb = None

            similarities = compute_similarity_batch(ref_img, images, ref_emb=ref_emb)
        except RuntimeError as e:
            # If model isn't available, return informative errors for similarity
            similarities = [None] * len(images)

    # Compute other metrics in parallel to speed up CPU-bound image processing
    def compute_metrics_pair(img):
        return {
            "color_accuracy": compute_color_accuracy(ref_img, img),
            "stroke_alignment": compute_stroke_alignment(ref_img, img),
            "composition_score": compute_composition_score(ref_img, img),
        }

    metrics_results = []
    if images:
        # use the same threadpool used for downloads
        futures = [ _download_executor.submit(compute_metrics_pair, img) for img in images ]
        metrics_results = [f.result() for f in futures]

    # Fill placeholders in results preserving input order
    j = 0
    for idx in valid_indices:
        sim = similarities[j] if j < len(similarities) else None
        metrics = metrics_results[j] if j < len(metrics_results) else {}
        results[idx] = {
            "url": payload.image_urls[idx],
            "similarity": sim,
            **metrics
        }
        j += 1

    return {"results": results}
