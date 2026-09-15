FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 构建机在境外（Render 等）时默认 PyPI 最快，在国内构建就传 --build-arg 换成清华源。
# 写死成任一边都会让另一边慢到怀疑人生。
ARG PIP_INDEX=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url ${PIP_INDEX} -r requirements.txt

# .dockerignore 挡掉了 data/ 和 .env，所以镜像里不会带上本机知识库和密钥
COPY . .

ENV PORT=8000
EXPOSE 8000

# 平台要塞哪个端口就用哪个：Render 会注入 PORT，本地不设时退回 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
