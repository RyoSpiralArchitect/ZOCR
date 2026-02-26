FROM python:3.12-slim

ARG ZOCR_APT_PACKAGES="poppler-utils"
ARG ZOCR_EXTRAS="api"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_DATA_HOME=/tmp/.local/share \
    ZOCR_API_STORAGE_DIR=/data \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN if [ -n "${ZOCR_APT_PACKAGES}" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends ${ZOCR_APT_PACKAGES} \
      && rm -rf /var/lib/apt/lists/*; \
    fi

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

COPY pyproject.toml README.md LICENSE LICENSING.md COMMERCIAL_LICENSE.md SECURITY.md CHANGELOG.md CITATION.cff ./
COPY zocr/ zocr/
COPY samples/ samples/

RUN python -m pip install --upgrade pip \
    && EXTRAS="$(echo "${ZOCR_EXTRAS}" | tr -d '[:space:]')" \
    && if [ -n "${EXTRAS}" ]; then python -m pip install ".[${EXTRAS}]"; else python -m pip install "."; fi \
    && groupadd -r zocr \
    && useradd -r -g zocr -m -d /home/zocr zocr \
    && mkdir -p /data \
    && chown -R zocr:zocr /app /data

USER zocr

EXPOSE 8000

CMD ["zocr", "--help"]
