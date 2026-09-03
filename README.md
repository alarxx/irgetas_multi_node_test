# Irgetas multi-node test

Minimal multi-node torchrun/NCCL `all_reduce` test.

---

```sh
module avail python # 3.11
module avail cuda # 12.6
```

Install uv:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Setup environment with uv:
```sh
uv python install 3.11
uv init --python 3.11
uv venv
uv pip install pip
# uv pip install setuptools wheel
. .venv/bin/activate
```

```sh
# uv add numpy
uv pip install numpy
```

---

Upgrade libs:
```sh
python -m pip install --upgrade pip 
python -m pip install --upgrade typing-extensions
# python -m pip install --upgrade setuptools wheel
# python -m pip install --upgrade build
# python -m pip install --upgrade twine
```

PyTorch 2.12 installation for CUDA 12.6:
```sh
uv pip install --force-reinstall torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 torchcodec==0.16.0 --index-url https://download.pytorch.org/whl/cu126
```

---

