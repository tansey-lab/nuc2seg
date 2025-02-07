FROM bitnami/tensorflow:latest
USER root
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git
RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install 'git+https://github.com/jeffquinn-msk/sopa.git#egg=sopa[stardist]'
RUN pip install --upgrade 'numpy<2'

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/bin/bash"]
