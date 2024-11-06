FROM bitnami/pytorch:latest

USER root
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install 'sopa[cellpose,baysor,tangram]'

ENV PYTHONUNBUFFERED=1
