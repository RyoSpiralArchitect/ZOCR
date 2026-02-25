FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE LICENSING.md COMMERCIAL_LICENSE.md SECURITY.md CHANGELOG.md CITATION.cff ./
COPY zocr/ zocr/
COPY samples/ samples/

ARG ZOCR_EXTRAS="api"
RUN python -m pip install --upgrade pip \
    && if [ -n "${ZOCR_EXTRAS}" ]; then python -m pip install ".[${ZOCR_EXTRAS}]"; else python -m pip install "."; fi

CMD ["zocr", "--help"]

