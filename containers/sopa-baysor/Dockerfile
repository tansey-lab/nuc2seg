FROM python:3.12

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 wget unzip git vim

RUN cd /tmp && wget 'https://github.com/kharchenkolab/Baysor/releases/download/v0.7.1/baysor-x86_x64-linux-v0.7.1_build.zip' && unzip baysor-x86_x64-linux-v0.7.1_build.zip && mv ./bin/baysor /opt/ && rm ./baysor-x86_x64-linux-v0.7.1_build.zip
ENV PATH="/opt/baysor/bin:${PATH}"

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install 'tensorflow'
RUN pip install 'git+https://github.com/jeffquinn-msk/sopa.git#egg=sopa[baysor]'

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/bin/bash"]
