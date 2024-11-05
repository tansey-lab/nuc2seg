FROM python:3.10

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install 'sopa[cellpose,baysor,tangram]'

ENV PYTHONUNBUFFERED=1
