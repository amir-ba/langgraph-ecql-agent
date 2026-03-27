FROM mtr.devops.telekom.de/community/python:3.12@sha256:2dabc7b4e421d7fef1ca495e65a127ec2c5bcdbd1d97fe42cd70c9e3963969b3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/home/cloud/.venv \
    PATH="/home/cloud/.venv/bin:${PATH}"

WORKDIR /home/cloud/app

# Install uv for reproducible dependency resolution from uv.lock.
RUN python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
RUN python -m pip install --no-cache-dir uv

# Copy dependency metadata first to maximize Docker layer caching.
COPY pyproject.toml uv.lock ./
RUN python -m uv sync --frozen --no-install-project
# run tests
# Copy the FastAPI application source.
COPY . .
RUN python -m uv run pytest --maxfail=1 --disable-warnings -q

EXPOSE 8080

# Avoid a runtime dependency on `python -m uv`; run uvicorn directly from the venv.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
