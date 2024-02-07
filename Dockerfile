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

#START insightface
RUN mkdir -p /root/.insightface/models/
RUN pip3 install -U insightface
#RUN wget -O /root/.insightface/models/inswapper_128.onnx https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
RUN wget -O /root/.insightface/models/inswapper_128.onnx https://huggingface.co/Devia/G/resolve/main/inswapper_128.onnx
RUN pip3 install onnxruntime-gpu
RUN python3 -c "import torch; import insightface; import onnxruntime; PROVIDERS = onnxruntime.get_available_providers(); [PROVIDERS.remove(provider) for provider in PROVIDERS if provider == 'TensorrtExecutionProvider']; insightface.app.FaceAnalysis(name='buffalo_l', providers=PROVIDERS)" || true
#END insightface

#ADD ./requirements.txt /app/ghost/
#RUN pip3 install -r requirements.txt

RUN pip3 install numpy
RUN pip3 install opencv-python
RUN pip3 install onnxruntime-gpu
RUN pip3 install onnx
RUN rm -rf /tmp/*
RUN apt-get clean
RUN pip3 install mxnet-cu101mkl
RUN pip3 install scikit-image
RUN pip3 install insightface
RUN pip3 install requests==2.25.1
RUN pip3 install kornia==0.5.4
RUN pip3 install dill
RUN pip3 install wandb

#ADD ./export-onnx.py /app/ghost/
ADD ./models/ ./models/
ADD ./network/ ./network/
ADD ./utils/ ./utils/
ADD ./AdaptiveWingLoss/ ./AdaptiveWingLoss/
ADD *.py .
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD ["bash"]