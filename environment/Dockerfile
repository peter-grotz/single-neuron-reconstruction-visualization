# Code Ocean compute environment.
#
# The base image must come from the Allen deployment's registry; Docker Hub
# images are rejected at launch with "invalid base image registry".
#
# The renderer is installed from the public repository at a pinned commit
# rather than from the capsule's own files, so a released capsule names exactly
# the code it ran and does not depend on where a git clone lays the package out.
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

ARG RENDERER_REF=6a2702b1558415e00e7b869fa47612f2fc1fdb7f
RUN pip install --no-cache-dir \
    "ccf-glass-render @ git+https://github.com/peter-grotz/single-neuron-reconstruction-visualization.git@${RENDERER_REF}"
