FROM python:3.12-slim

WORKDIR /app

# LightGBM links against the OpenMP runtime, which the slim base image omits.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --locked --no-dev

COPY app app
COPY configs/features.yaml configs/features.yaml
COPY src src
COPY artifacts/release artifacts/release

EXPOSE 8501

CMD ["uv", "run", "--no-dev", "--no-sync", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
