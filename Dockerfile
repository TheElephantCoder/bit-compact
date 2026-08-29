# syntax=docker/dockerfile:1.4
FROM rust:1.77-bookworm AS builder
WORKDIR /app

# cache deps first (no deps here, but keep pattern for future)
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY benches ./benches
COPY examples ./examples
# build release binary (bitcompact CLI + library)
RUN cargo build --release --bin bitcompact && strip target/release/bitcompact && ls -lh target/release/bitcompact

FROM debian:bookworm-slim AS runtime
LABEL org.opencontainers.image.source="https://github.com/TheElephantCoder/bit-compact"
LABEL org.opencontainers.image.description="bit-compact — SQ8 embedding compression, 4× smaller, 1 seek, Docker GUI"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.vendor="TheElephantCoder"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini && rm -rf /var/lib/apt/lists/*
WORKDIR /data

COPY --from=builder /app/target/release/bitcompact /usr/local/bin/bitcompact
RUN bitcompact --help || true

EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["tini", "--", "bitcompact"]
CMD ["--help"]
