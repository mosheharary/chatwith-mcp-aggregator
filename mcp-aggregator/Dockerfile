FROM python:3.13.2-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm (required for uvx and MCP server dependencies)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Verify installations
RUN python --version \
    && node --version \
    && npx --version

# Install uv for fast Python package management
RUN pip install uv
RUN uv --version && uvx --version

# Set working directory
WORKDIR /app

# Copy application files
COPY main.py /app/main.py
COPY server_manager.py /app/server_manager.py
COPY requirements.txt /app/requirements.txt
COPY servers_config.json /app/servers_config.json

# Install Python dependencies (including FastMCP)
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose FastMCP HTTP server port
EXPOSE 8000

# Add healthcheck for the FastMCP server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/mcp || exit 1

# Set metadata labels
LABEL description="FastMCP HTTP server aggregating multiple MCP backends"
LABEL version="2.0"
LABEL maintainer="MCP Aggregator"

# Run the FastMCP aggregator server
CMD ["python", "main.py"]
