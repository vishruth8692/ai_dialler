FROM python:3.11-slim

# build-essential: chromadb's hnswlib dependency sometimes needs to compile from source if no
# prebuilt wheel matches the target platform.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first - the default PyPI wheel bundles full NVIDIA CUDA runtime packages
# (~2.7GB of nvidia-cublas/cudnn/nccl/triton/etc.) that are useless on Railway's CPU-only
# builders and are almost certainly why the previous deploy failed (bloated multi-GB image,
# likely OOM-killed on container start with no time to log anything). Installing this first means
# pip sees torch as already satisfied and never pulls the GPU build when it resolves
# sentence-transformers' dependency on it below.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the multilingual embedding model into the image at build time - qa_store.warm_up() (see
# app/main.py startup hook) still runs on every process start, but this avoids it having to hit
# Hugging Face Hub over the network each time, which added several seconds to every local restart.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

COPY app ./app

# Railway (and most PaaS hosts) inject $PORT at runtime - bind to it, not a hardcoded port.
# JSON/exec form (with an explicit shell) so SIGTERM forwards correctly on redeploy/shutdown,
# while still allowing the ${PORT:-8008} shell substitution uvicorn's own args can't do alone.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8008}"]
