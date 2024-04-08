cd "$(dirname "$0")"
git pull >> logx.txt 2>&1
docker run -u "$(id -u):$(id -g)" -v $PWD:/app --workdir /app ghcr.io/getzola/zola:v0.18.0 build  >> logx.txt 2>&1
docker compose restart  >> logx.txt 2>&1
