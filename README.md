# Irgetas multi-node test

Minimal multi-node [Slurm](https://slurm.schedmd.com/quickstart.html)/torchrun/NCCL `all_reduce` test.

Irgetas is a multi-node H100 cluster (https://hpc.nu.edu.kz/).

---

The repository contains sample code for synchronizing two nodes, each with 4 GPUs:
```
g001:
    torchrun
      ├── rank 0 → GPU0
      ├── rank 1 → GPU1
      ├── rank 2 → GPU2
      └── rank 3 → GPU3

g002:
    torchrun
      ├── rank 4 → GPU0
      ├── rank 5 → GPU1
      ├── rank 6 → GPU2
      └── rank 7 → GPU3
```

---

## Setup

```sh
module avail python # 3.11
module avail cuda # 12.6
```

Install [uv](https://docs.astral.sh/uv/):
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

[PyTorch 2.12](https://pytorch.org/get-started/previous-versions/) installation for CUDA 12.6:
```sh
uv pip install --force-reinstall torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 torchcodec==0.16.0 --index-url https://download.pytorch.org/whl/cu126
```

---

## License

[MIT](https://github.com/alarxx/irgetas_multi_node_test/blob/main/LICENSE)
