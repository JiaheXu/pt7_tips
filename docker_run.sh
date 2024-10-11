
xhost +local:root
DATA_PATH=~/
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $DATA_PATH:/ws \
    --network=host --name skills -it 4766ab79dd80



xhost +local:root
DATA_PATH=~/
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $DATA_PATH:/ws \
    --network=host --name CSIRL -it jiahexu98/skills:latest



xhost +local:root
DATA_PATH=~/
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $DATA_PATH:/ws \
    --network=host --name skill2 -it pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    
    
FOLDER=/home/jiahexu
sudo docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $FOLDER:/workspace \
    --network=host -it --name torch1.9 pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
    
    
FOLDER=/home/jiahexu

xhost +local:root
FOLDER=/home/xujiahe
sudo docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $FOLDER:/workspace \
    --network=host -it --name stitching pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
 
 
FOLDER=/home/jiahexu
xhost +local:root
FOLDER=/home/xujiahe
sudo docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $FOLDER:/workspace \
    --network=host -it --name retarget pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
 
FOLDER=/home/mmpug/ws
sudo docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $FOLDER:/workspace \
    --network=host -it --name IsaacGym puzlcloud/pytorch:1.10.1-cuda11.3-cudnn8-jupyter-g1-1.1.0-python3.8
    
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
apt-get update
apt-get upgrade

apt-get install git

xhost +local:root
FOLDER=/home/xujiahe
sudo docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $FOLDER:/workspace \
    --network=host -it --name frankmocap frankmocap/cuda10.1:latest 
    
    
    
    
    
    
    
    
pip install scipy Cython
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric==2.0.1
