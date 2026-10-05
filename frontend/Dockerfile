# Frontend image.
#
# Two stages: build the static bundle with Node, then serve it from
# nginx. The runtime image contains no Node and no source.
FROM node:22-alpine AS builder

WORKDIR /app

# Installed from the lockfile so builds are reproducible.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]