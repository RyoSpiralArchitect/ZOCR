FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ZOCR_API_STORAGE_DIR=/data \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

COPY pyproject.toml README.md LICENSE LICENSING.md COMMERCIAL_LICENSE.md SECURITY.md CHANGELOG.md CITATION.cff ./
COPY zocr/ zocr/
COPY samples/ samples/

ARG ZOCR_EXTRAS="api"
RUN python -m pip install --upgrade pip \
    && if [ -n "${ZOCR_EXTRAS}" ]; then python -m pip install ".[${ZOCR_EXTRAS}]"; else python -m pip install "."; fi \
    && groupadd -r zocr \
    && useradd -r -g zocr -m -d /home/zocr zocr \
    && mkdir -p /data \
    && chown -R zocr:zocr /app /data

USER zocr

EXPOSE 8000

CMD ["zocr", "--help"]
