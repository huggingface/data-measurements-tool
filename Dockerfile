# Start from a Python image
FROM python:3.9.6-slim-bullseye

# Set working directory
WORKDIR /app

# Update OS packages and install essential tools for Rust
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Set environment variable for Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Update pip and install python dependencies
RUN pip install --upgrade pip
COPY requirements_ezi.txt /app
RUN pip install  --prefer-binary --no-cache-dir -r requirements_ezi.txt
RUN apt-get update && \
    apt-get install -y sudo dbus && \
    rm -rf /var/lib/apt/lists/*