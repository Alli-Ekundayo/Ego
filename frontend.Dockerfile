# Build stage
FROM node:20-slim AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN rm -f package-lock.json && npm install
COPY frontend/ ./
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html/

# Ensure the data directory exists (mounted at runtime)
RUN mkdir -p /usr/share/nginx/html/data

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/index.html || exit 1

CMD ["nginx", "-g", "daemon off;"]