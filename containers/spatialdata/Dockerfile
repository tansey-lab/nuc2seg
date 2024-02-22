FROM python:3.10

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install spatialdata-io pyarrow pandas ipython

RUN mkdir -p /app

COPY create_sd.py /app/

ENV PYTHONUNBUFFERED=1

WORKDIR /app
