FROM mtr.devops.telekom.de/community/python:3.12@sha256:2dabc7b4e421d7fef1ca495e65a127ec2c5bcdbd1d97fe42cd70c9e3963969b3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv for reproducible dependency resolution from uv.lock.
RUN python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
RUN python -m pip install --no-cache-dir uv

# Copy dependency metadata first to maximize Docker layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the FastAPI application source.
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
