FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && apt-get upgrade -y && apt-get install -y vim && apt-get install -y git
RUN pip install --upgrade pip

