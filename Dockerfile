# syntax=docker/dockerfile:1.4
FROM rust:1.77-bookworm AS builder
WORKDIR /app

# cache deps first (no deps here, but keep pattern for future)
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY examples ./examples
# build release binary (bitcompact CLI + library)
RUN cargo build --release --bin bitcompact && strip target/release/bitcompact && ls -lh target/release/bitcompact

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini && rm -rf /var/lib/apt/lists/*
WORKDIR /data

COPY --from=builder /app/target/release/bitcompact /usr/local/bin/bitcompact
# ensure binary is executable and check
RUN bitcompact --help || true

EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["tini", "--", "bitcompact"]
CMD ["--help"]
