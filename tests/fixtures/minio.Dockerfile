FROM golang:1.24.2 AS build
RUN git clone --depth 1 --branch RELEASE.2025-04-22T22-12-26Z https://github.com/minio/minio.git /src
WORKDIR /src
RUN test "$(git rev-parse HEAD)" = "0d7408fc9969caf07de6a8c3a84f9fbb10a6739e"
RUN go mod download
RUN CGO_ENABLED=0 go build -trimpath -o /minio .
FROM scratch
COPY --from=build /minio /minio
EXPOSE 9000
ENTRYPOINT ["/minio"]
CMD ["server", "/data", "--address", ":9000"]
