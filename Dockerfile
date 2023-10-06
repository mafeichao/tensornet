FROM ubuntu:18.04

RUN apt-get install python3.7 python3.7-dev -y && \
    apt-get update && apt-get install wget vim -y && \
    apt-get install python3-pip -y && \
    apt-get install git -y && \
    apt-get install libssl-dev -y && \
    apt-get install lib32z1-dev -y && \
    apt-get install curl -y

RUN wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-linux-x86_64 && \
    mv bazel-3.1.0-linux-x86_64 /usr/local/bin/bazel && \
    chmod a+x /usr/local/bin/bazel

RUN ln -s /usr/bin/python3.7 /usr/bin/python

RUN python -m pip install --upgrade pip && \
    python -m pip install tensorflow==2.3.0

RUN wget https://download.open-mpi.org/release/open-mpi/v1.4/openmpi-1.4.5.tar.gz && \
    mkdir -p /root/opt && \
    tar -zxf openmpi-1.4.5.tar.gz -C /root/opt/ && \
    mv /root/opt/openmpi-1.4.5 /root/opt/openmpi && \
    cd /root/opt/openmpi && \
    ./configure CFLAGS="-fPIC" CXXFlAGS="-fPIC" --prefix=/root/opt/openmpi --enable-static && \
    make -j20 && \
    make install

WORKDIR /tensornet
COPY . .
#RUN git clone https://github.com/Qihoo360/tensornet.git && \
#    cd /tensornet && \
RUN bash configure.sh --openmpi_path /root/opt/openmpi && \
    bazel build -c opt //core:_pywrap_tn.so && \
    cp -f /tensornet/bazel-bin/core/_pywrap_tn.so /tensornet/tensornet/core

ENV PATH "/root/opt/openmpi/bin:${PATH}"
ENV PYTHONPATH "/tensornet:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/root/opt/openmpi/lib:${LD_LIBRARY_PATH}"

CMD ["python", "-c", "import tensorflow as tf; import tensornet as tn; print(tn.version)"]
