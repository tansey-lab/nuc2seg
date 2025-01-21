FROM bitnami/tensorflow:latest

USER root

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython memray stardist gputools
RUN --mount=type=cache,target=/.cache/pip pip install --upgrade 'numpy<2'
RUN --mount=type=cache,target=/.cache/pip pip install sopa
ENV PYTHONUNBUFFERED=1

ENTRYPOINT []
