# GPT-SoVITS-V3-Infer-API

#### 🎉Now you can infer with api for v3 model!

This is the inference code of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) that can be developer-friendly

The project code is modified from [GPT-SoVITS-Infer](https://github.com/BeautyyuYanli/GPT-SoVITS-Infer).

Compatible with **all version** model calls.

## Usage Example

See `api-v3.py`.

See [api-example.py](example.ipynb) or [main.py]() example code.

You must set model path in `api-config.yaml` file!

## Prepare the environment

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
```

</details>

<details><summary>Windos</summary>


```
pdm install
```

</details>

After the deployment is complete, you can go to `api-v3.py` to check the specific usage methods. 

There are relevant request example codes in `api-example.py` that you can refer to.

