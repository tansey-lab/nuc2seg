FROM bitnami/pytorch:2.2.0-debian-11-r1

USER root

RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/.cache/pip pip install ipython

RUN mkdir -p /app
COPY src/ /app/src/

COPY pyproject.toml setup.py LICENSE.txt README.md /app/

RUN --mount=type=cache,target=/.cache/pip cd /app && pip install '.[dev,test]'

ENV PYTHONUNBUFFERED=1

WORKDIR /app
