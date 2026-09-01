# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
# ---------------------------------------------------------------------------
# 1. System dependencies + Python
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        wget git build-essential ca-certificates \
        && rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------------------------
# 2. Install PyTorch (CUDA 12.8), xformers, and ninja
# ---------------------------------------------------------------------------
RUN pip3 install  --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
# ---------------------------------------------------------------------------
# 3. Install project dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        numpy omegaconf msgspec rouge-score sacrebleu \
        sentencepiece tiktoken blobfile wandb viztracer \
        lm-eval scipy pynvml datatrove orjson \
        transformers==5.10.2 tokenizers datasets==4.7.0 tyro==1.0.13 \
        torchdata==0.11.0 einops tensorboard
RUN pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130
RUN pip install torch vllm torchcomms  --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu130

# ---------------------------------------------------------------------------
# 4. AWS EFA / OFI-NCCL (aligned with production Dockerfile overlay)
# ---------------------------------------------------------------------------
ARG AWS_EFA_INSTALLER_URL=https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
ENV PATH=/opt/amazon/efa/bin:${PATH} \
        LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/ofi-nccl/lib:/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        iproute2 \
        cpio \
        linux-headers-aws \
        linux-aws \
        libhwloc15 \
        tar \
        unzip \
        pciutils environment-modules tcl \
        && rm -rf /var/lib/apt/lists/* \
        || (apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        cpio \
        iproute2 \
        libhwloc15 \
        tar \
        unzip \
        pciutils environment-modules tcl \
        && rm -rf /var/lib/apt/lists/*) \
        || true
RUN curl -fsSL -o aws-efa-installer-latest.tar.gz "${AWS_EFA_INSTALLER_URL}"
RUN tar --no-same-owner -xf aws-efa-installer-latest.tar.gz
RUN apt-get update && \
        apt-get install -y linux-aws linux-headers-aws || true
RUN cd aws-efa-installer && \
        ./efa_installer.sh -y --skip-kmod --disable-ngc --skip-mpi
RUN plugin_path="$(find /opt/amazon /opt/aws-ofi-nccl -name 'libnccl-net.so' -print -quit 2>/dev/null || true)" \
        && test -n "${plugin_path}" \
        && ldconfig \
        && for required_lib in libhwloc.so.15 libfabric.so.1 libefa.so.1 libibverbs.so.1; do \
        ldconfig -p | grep -q "${required_lib}" || exit 1; \
        done \
        && ldd "${plugin_path}" \
        && ! ldd "${plugin_path}" | grep -q "not found" \
        && echo "Found NCCL net plugin at ${plugin_path}"
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
        unzip -q /tmp/awscliv2.zip -d /tmp && \
        /tmp/aws/install && \
        rm -rf /tmp/awscliv2.zip /tmp/aws
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt install -y zstd && apt install -y tmux && apt install -y unzip

# Ensure the container runs as root
USER root

CMD ["bash"]
