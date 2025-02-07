FROM bitnami/pytorch:latest
USER root
RUN apt-get update && apt-get install -y git
RUN --mount=type=cache,target=/.cache/pip pip install --upgrade pip
RUN pip install 'git+https://github.com/jeffquinn-msk/sopa.git#egg=sopa[cellpose]'
COPY extract_shapefile_from_sopa.py /app/

RUN chmod +x /app/extract_shapefile_from_sopa.py

RUN ln -s /app/extract_shapefile_from_sopa.py /usr/local/bin/extract_shapefile_from_sopa

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/bin/bash"]
