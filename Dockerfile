# ──────────────────────────────────────
# 1) Build React frontend
# ──────────────────────────────────────
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build
# build output is in /app/frontend/build

# ──────────────────────────────────────
# 2) Final image: Python + Static only
# ──────────────────────────────────────
FROM python:3.11-slim

# Install system deps: curl, bash, netcat-openbsd, git
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      bash \
      ca-certificates \
      gnupg \
      netcat-openbsd \
      git \
 && rm -rf /var/lib/apt/lists/*

# Install Node.js for `serve`
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g serve \
 && rm -rf /var/lib/apt/lists/*

# Prepare workdir
WORKDIR /app/backend

# Python deps
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + React build
COPY backend/ ./
COPY --from=frontend-build /app/frontend/build ./static

# Entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose ports
EXPOSE 6146 3000 6379

# Entrypoint
ENTRYPOINT ["entrypoint.sh"]
