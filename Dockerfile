FROM python:3.11-slim

WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 步骤1：先单独安装 torch CPU 版（约 200MB，远小于默认 CUDA 版 526MB）。
# 使用 --no-deps 避免从 PyTorch CDN 拉取 torch 的依赖（CDN 对非 torch 包很慢），
# 依赖全部留给步骤2 从清华镜像源拉取（速度快）。
RUN pip install --no-cache-dir --no-deps --default-timeout=600 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

# 步骤2：复制依赖文件并安装其余依赖（从清华镜像源加速）。
# 这里会装 torch 的依赖（sympy/networkx/jinja2 等）以及 requirements.txt 中的全部包。
# 已装好的 torch 不会重复下载；sentence-transformers 会复用现有 torch。
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
