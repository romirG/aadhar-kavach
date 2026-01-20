# UIDAI Express Server Dockerfile
# Node.js API Gateway and Static File Server

FROM node:20-alpine

# Create app directory
WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodeuser -u 1001

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY --chown=nodeuser:nodejs . .

# Switch to non-root user
USER nodeuser

# Expose port
EXPOSE 3001

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:${PORT:-3001}/api/health || exit 1

# Start the server
CMD ["node", "index.js"]
