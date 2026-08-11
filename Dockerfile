# CPU image. For GPU scoring, swap the base for an nvidia/cuda runtime and
# install the matching torch wheel instead of the CPU index below.
#
# Not the -slim variant: Triton compiles kernels at runtime for some
# architectures (Llama among them) and needs the toolchain that slim omits.
FROM python:3.12

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across code changes.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python - <<'PY' > /tmp/reqs.txt
import tomllib
with open("pyproject.toml", "rb") as f:
    project = tomllib.load(f)["project"]
deps = [d for d in project["dependencies"] if not d.startswith("torch")]
deps += project["optional-dependencies"]["chronogpt"]
print("\n".join(deps))
PY
RUN pip install --no-cache-dir -r /tmp/reqs.txt

COPY wigin_tllm/ wigin_tllm/
RUN pip install --no-cache-dir --no-deps -e .

# Cache and SQLite live here; mount a volume to keep them across runs.
ENV WIGIN_TLLM_DATA_DIR=/state
VOLUME ["/state"]

ENTRYPOINT ["wigin-tllm"]
CMD ["--help"]
