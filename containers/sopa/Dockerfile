FROM bitnami/pytorch:latest

USER root
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 wget unzip git vim

RUN cd /tmp &&  wget 'https://github.com/kharchenkolab/Baysor/releases/download/v0.7.1/baysor-x86_x64-linux-v0.7.1_build.zip' && unzip baysor-x86_x64-linux-v0.7.1_build.zip && mv ./bin/baysor /opt/ && rm ./baysor-x86_x64-linux-v0.7.1_build.zip
ENV PATH="/opt/baysor/bin:${PATH}"

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install 'tensorflow'
RUN pip install 'git+https://github.com/jeffquinn-msk/sopa.git#egg=sopa[cellpose,baysor,tangram,stardist]'
RUN pip install --upgrade 'numpy<2'

COPY extract_shapefile_from_sopa.py /app/

RUN chmod +x /app/extract_shapefile_from_sopa.py

RUN ln -s /app/extract_shapefile_from_sopa.py /usr/local/bin/extract_shapefile_from_sopa

ENV PYTHONUNBUFFERED=1
