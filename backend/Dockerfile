# Use lightweight Ubuntu base
FROM ubuntu:latest

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install compilers and runtimes
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    default-jdk \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code
