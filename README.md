# Image-Scoring-Service-

Image Scoring Service

## Deploying to Render

Quick steps to deploy this FastAPI app on Render (managed service):

1. Push this repository to GitHub.
2. Go to https://render.com and create a new Web Service. Connect your GitHub repo.
3. Select the `main` branch (or the branch you pushed).
4. For the Build Command use:

	pip install -r requirements.txt

5. For the Start Command use (Render provides $PORT automatically):

	uvicorn main:app --host 0.0.0.0 --port $PORT

Notes and troubleshooting:

- This project depends on heavy packages (torch, sentence-transformers, opencv). The build can take several minutes and may fail on small instances due to memory/disk limits. If you encounter build failures, consider one of the following:
  - Use a Render service with a larger instance plan.
  - Build a Docker image locally (or via a CI) that installs optimized/wheel builds for torch and push it to a container registry, then deploy a Docker service on Render.
  - For quick testing, trim requirements to a minimal FastAPI app and add the heavy libraries later.

- Use `opencv-python-headless` in `requirements.txt` to avoid GUI dependencies on server.
- If you need GPU acceleration, Render does not provide GPU instances; consider a cloud provider that supports GPU or use a managed inference provider.

If you'd like, I can:

- Add a `render.yaml` manifest for infra-as-code.
- Add a small Dockerfile that installs CPU-only torch wheels to improve build reliability.
# Image-Scoring-Service-
Image Scoring Service 
