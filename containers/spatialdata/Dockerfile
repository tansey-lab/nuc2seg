FROM python:3.10

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install spatialdata-io pyarrow pandas ipython

RUN mkdir -p /app

COPY create_sd.py /app/

RUN chmod +x /app/create_sd.py

RUN ln -s /app/create_sd.py /usr/local/bin/create_sd

ENV PYTHONUNBUFFERED=1

WORKDIR /app
