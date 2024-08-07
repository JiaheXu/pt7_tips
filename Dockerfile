# Use an official Python runtime as a parent image
# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04

# Set the working directory to /app
WORKDIR /ws

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install python3.8 python3.8-dev python3.8-venv python3-pip python3-tk python-is-python3 -y && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
SHELL ["/bin/bash", "--login", "-c"]

RUN pip install --upgrade pip
RUN pip install pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch_geometric
RUN pip install fvcore iopath
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html
# RUN pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html
# RUN pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

# RUN mkdir /code
# WORKDIR /code

# RUN mkdir equivariant_pose_graph
# WORKDIR /code/equivariant_pose_graph

# COPY configs/ ./configs/
# COPY python/ ./python/
# COPY scripts/ ./scripts/
# COPY setup.py .
# COPY README.md .
RUN apt update && apt install libgl1-mesa-glx graphviz -y

RUN pip install pytorch-lightning==1.9.0
RUN pip install wandb
RUN pip install matplotlib
RUN pip install colour
RUN pip install imageio
RUN pip install hydra-core
RUN pip install https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.6.0-cp38-cp38-manylinux_2_28_x86_64.whl
