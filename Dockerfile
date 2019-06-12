FROM mcr.microsoft.com/cntk/release:2.6-cpu-python3.5

WORKDIR /project

COPY train.py .
COPY ResNet18_ImageNet_CNTK.model .
ADD data ./data

CMD ["bash ","-c 'source /cntk/activate-cntk && train.py'"]