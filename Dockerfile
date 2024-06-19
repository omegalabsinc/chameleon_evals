FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get remove -y python3 && apt autoremove -y && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    ca-certificates \
    curl \
    wget \
    unzip \
    rustc \
    cargo \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3 and python
RUN ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python

# get pip without grabbing python3.8 as a dependency
WORKDIR /build
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

CMD ["/bin/bash"]