# Code Ocean compute environment.
#
# The renderer is installed from the public repository at a pinned commit rather
# than from the capsule's own files, so a released capsule names exactly the
# code it ran and does not depend on where a git clone lays the package out.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

ARG RENDERER_REF=d4df949b3d5262736858a943ce0e3e2344fc8eda
RUN pip install --no-cache-dir \
    "ccf-glass-render @ git+https://github.com/peter-grotz/single-neuron-reconstruction-visualization.git@${RENDERER_REF}"
