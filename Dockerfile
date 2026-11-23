# Headless Qwen3-VL batch captioner for Runpod Serverless.
# No model weights are baked in: Qwen3-VL-4B-Instruct (~8.27 GiB) is downloaded
# at job start and cached on the network volume (/runpod-volume/qwen-vl/models)
# when one is attached.
#
# Official Qwen3-VL code has no tags; the SHA below is documented for
# attribution and the official README/API. Inference runs through a released
# transformers pin (not a git clone of floating main, and not git-main of
# transformers). Our GitHub release for the Hub must use targetCommitish=main.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG QWEN3_VL_SHA=96588727e44c78b25ba03ea03b8e12f7e64fd0da
ARG TRANSFORMERS_VERSION=4.57.6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    QWEN3_VL_SHA=${QWEN3_VL_SHA} \
    MODEL_ID=Qwen/Qwen3-VL-4B-Instruct \
    MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17

# python3.12 is the Ubuntu 24.04 default (transformers 5 would also need >=3.11).
# ffmpeg: exotic image containers. git is not required at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg \
        python3 python3-venv python3-pip python3-dev \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"

# Record the documented upstream SHA (code pin for README / attribution).
RUN printf '%s\n' "${QWEN3_VL_SHA}" > /opt/qwen3-vl.commit \
    && test "$(cat /opt/qwen3-vl.commit)" = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"

# Use a venv so we do not fight the Debian pip install (no RECORD file).
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Stage 1: torch 2.7.1+cu128 from download.pytorch.org, same pin as the other
# workers. Do not upgrade torch after this.
# Stage 2: a released transformers >= 4.57.0 (Qwen3-VL landed in 4.57.0).
# The model card still says "install git-main, 4.57.0 is not released" — that
# is stale. Do not install huggingface/transformers from a floating git URL.
WORKDIR /opt
RUN pip install --upgrade pip \
    && pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt
RUN python -m pip install \
        transformers==${TRANSFORMERS_VERSION} \
        accelerate==1.10.1 \
        pillow==11.2.1 \
        -c /tmp/constraints.txt \
    && python -c "import transformers; from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForImageTextToText; assert transformers.__version__=='${TRANSFORMERS_VERSION}'" \
    && python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && python -m pip freeze > /app/requirements.lock \
    && python -c "import importlib.metadata as m,sys,torch,transformers,accelerate,PIL,runpod,qwen_vl_utils; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__, '| python', sys.version.split()[0], '| runpod', runpod.__version__, '| qwen-vl-utils', m.version('qwen-vl-utils'), '| qwen3-vl sha', open('/opt/qwen3-vl.commit').read().strip())"

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
# Gradio / official web demo / vLLM are never installed or started.
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
