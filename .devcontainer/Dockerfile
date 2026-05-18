FROM mcr.microsoft.com/devcontainers/python:3.12

# Install UV from official image — avoids shell installer and keeps PATH clean
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
