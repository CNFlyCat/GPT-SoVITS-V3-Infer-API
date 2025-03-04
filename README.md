# GPT-SoVITS-V3-Infer-API

#### 🎉Now you can infer with api for v3 model!

This is the inference code of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) that can be developer-friendly

The project code is modified from [GPT-SoVITS-Infer](https://github.com/BeautyyuYanli/GPT-SoVITS-Infer).

Compatible with **all version** model calls.

---

## Usage Example

See `api-v3.py`.

See [api-example.py](example.ipynb) or [main.py]() example code.

---

## Prepare the environment

You must download **chinese-hubert-base**, **chinese-roberta-wwm-ext-large**, and **models--nvidia--bigvgan_v2_24khz_100band_256x** from [HuggingFace(this url)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) and **place them** in the **pretrained_models** folder!

You must **set model path** in **api-config.yaml** file!

You can use PDM to install this project quickly.

If you want to run it on CPU, you must remove this in `pyproject.toml`.

```toml
# remove this
[[tool.pdm.source]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu118"
include_packages = ["torch","torchaudio","torchvision"]
exclude_packages = ["*"]
```

<details><summary> Linux</summary>

```
pdm install
python api-v3.py
```

</details>

<details><summary>Windos</summary>

```
pdm install
python api-v3.py
```

</details>

After the deployment is complete, you can go to `api-v3.py` to check the specific usage methods.

There are relevant request example codes in `api-example.py` that you can refer to.

> **If you encounter any problems or have any good suggestions during use, you can raise them on Issues and I will try my best to solve them.**

---
