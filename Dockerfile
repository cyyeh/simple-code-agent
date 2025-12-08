FROM docker.io/python:3.12.12-slim-bookworm AS base

# 1. Environment & basic settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 2. Install system dependencies (needed for geopandas, prophet, etc.)
#    - build-essential: compilers for building some wheels from source
#    - gdal, proj, geos: geospatial stack for geopandas
#    - curl, ca-certificates: good to have for network access / TLS
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        gcc \
        gfortran \
        libgdal-dev \
        gdal-bin \
        libproj-dev \
        proj-bin \
        libgeos-dev \
        libgeos++-dev \
        libspatialindex-dev \
        libxml2-dev \
        libxslt1-dev \
        liblapack-dev \
        libblas-dev \
        git \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 3. Optional: create a dedicated directory and (if you like) a venv
WORKDIR /app

# If you prefer a virtualenv instead of installing into the global site-packages:
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# 4. Upgrade pip once
RUN python -m pip install --upgrade pip

# 5. Install Python dependencies in a single layer
RUN pip install --no-cache-dir \
    requests \
    pandas \
    numpy \
    geopandas \
    matplotlib \
    seaborn \
    folium \
    wordcloud \
    scipy \
    scikit-learn \
    prophet \
    arch \
    sympy \
    pm4py \
    thefuzz \
    reportlab

# 6. Default command (you can override in docker run / compose)
CMD ["python"]