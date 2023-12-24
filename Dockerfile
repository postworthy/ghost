from nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install -y python3 pip python3-tk wget unzip git
RUN apt-get install -y libgl1-mesa-glx libgtk-3-0

RUN pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/postworthy/ghost

WORKDIR /app/ghost

RUN sh download_models.sh

RUN git submodule init
RUN git submodule init

ADD ./requirements.txt /app/ghost/
RUN pip3 install -r requirements.txt


ADD ./export-onnx.py /app/ghost/
ADD . .
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["bash"]