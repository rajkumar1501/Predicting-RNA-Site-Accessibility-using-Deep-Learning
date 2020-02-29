# hash:sha256:2d7c491fd00484fbd1f007023ba7beb7a53669e0356d136d3d0924ee47a4e988
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        keras==2.2.4 \
        pydot==1.4.1 \
        scikit-learn==0.21.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorflow-gpu==2.0.0
