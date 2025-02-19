"""
# WebAPI文档

` python api-v3.py -a 127.0.0.1 -p 9880 `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
### 配置文件:
    在 api-config.yaml 配置需要加载的模型路径

### 示例：
    在 api-example 里有调用示例代码
    在 main 当中叶游示例

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&media_type=wav&streaming_mode=True
```

可用参数：
Parameters:

text: str = None,                   # str.(required) text to be synthesized
text_lang: str = "auto",            # str.(required) language of the text to be synthesized
ref_audio_path: str = None,         # str.(required) reference audio path
prompt_lang: str = None,            # str.(required) language of the prompt text for the reference audio
prompt_text: str = "",              # str.(optional) prompt text for the reference audio
top_k: int = 5,                     # int. top k sampling
top_p: float = 1,                   # float. top p sampling
temperature: float = 1,             # float. temperature for sampling
sample_steps: int = 16,             # int. When you use v3 model,you can set this sample_steps
media_type: str = "wav",            # str. Set the file format for returning audio.
streaming_mode: bool = False,       # bool. whether to return a streaming response.
threshold: int = 30                 # int. Text segmentation parameter,the lower value, the faster the streaming inference, but the worse the audio quality.


RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import argparse
import os
import signal
import sys
import traceback
import yaml
import uvicorn
import subprocess
import numpy as np
import soundfile as sf

from io import BytesIO
from typing import Generator
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
from gpt_sovits.infer import GPTSoVITSInference

parser = argparse.ArgumentParser(description="GPT-SoVITS-api")
parser.add_argument("-p", "--port", type=int, default=9880, help="server port")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="server ip")
args = parser.parse_args()


def load_config():
    config_path = "./api-config.yaml"
    default_config = {
        "device": "cuda",
        "is_half": False,
        "version": "v2",  # model version, keep it right
        "t2s_weights_path": r"pretrained_models/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": r"pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "bert_base_path": r"pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": r"pretrained_models/chinese-hubert-base",
    }

    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Config is create!")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("load config success.")
    return config


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def check_params(text: str = None,
                 text_lang: str = "auto",
                 ref_audio_path: str = None,
                 prompt_lang: str = None,
                 prompt_text: str = "",
                 media_type: str = "wav",
                 streaming_mode: bool = False,
                 ):
    if text_lang == "auto":
        print("Warning: you text_lang is auto!")
    if ref_audio_path in [None, ""]:
        print("Error: ref_audio_path is empty！")
        return JSONResponse(status_code=400, content={"Error": "ref_audio_path is empty！"})
    if text in [None, ""]:
        print("Error: text is empty！")
        return JSONResponse(status_code=400, content={"Error": "text is empty！"})
    if prompt_text in [None, ""]:
        print("Error: prompt_text is empty！")
        return JSONResponse(status_code=400, content={"Error": "prompt_text is empty！"})
    if prompt_lang in [None, ""]:
        print("Error: prompt_lang is empty！")
        return JSONResponse(status_code=400, content={"Error": "prompt_lang is empty！"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"Error": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"Error": "ogg format is not supported in non-streaming mode"})
    return None


def streaming_generator(tts_generator: Generator, media_type: str):
    for sr, chunk in tts_generator:
        yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()


config = load_config()
inference = GPTSoVITSInference(config.get("bert_base_path"),
                               config.get("cnhuhbert_base_path"),
                               config.get("device"),
                               config.get("is_half")
                               )

inference.load_sovits(config.get("vits_weights_path"), config.get("version", "v2"))
inference.load_gpt(config.get("t2s_weights_path"))

port = args.port
host = args.bind_addr
argv = sys.argv

APP = FastAPI()


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        inference.load_gpt(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        inference.load_sovits(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/tts")
async def tts_get_endpoint(
        text: str = None,
        text_lang: str = "auto",
        ref_audio_path: str = None,
        prompt_lang: str = None,
        prompt_text: str = "",
        top_k: int = 5,
        top_p: float = 1,
        temperature: float = 1,
        sample_steps: int = 16,
        media_type: str = "wav",
        streaming_mode: bool = False,
        threshold: int = 30
):
    check_res = check_params(text, text_lang, ref_audio_path, prompt_lang, prompt_text, media_type, streaming_mode)
    if check_res is not None:
        return check_res

    try:
        inference.set_prompt_audio(prompt_text, prompt_lang, ref_audio_path)
        version = config.get("version", "v2")

        if streaming_mode:
            print("Streaming_mode...")
            tts_generator = inference.get_tts_wav_stream(text, text_lang, top_k, top_p,
                                                         temperature, sample_steps, version, threshold)
            sample_rate = 24000 if version == "v3" else 32000
            headers = {"sample_rate": str(sample_rate)}
            return StreamingResponse(streaming_generator(tts_generator, "raw"),
                                     headers=headers, media_type=f"audio/{media_type}")
        else:
            sample_rate, audio_data = inference.get_tts_wav(text, text_lang, top_k, top_p,
                                                            temperature, sample_steps, version)
            audio_data = pack_audio(BytesIO(), audio_data, sample_rate, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")

    except Exception as e:
        return JSONResponse(status_code=400, content={"Error": f"tts failed", "Exception": str(e)})


if __name__ == "__main__":
    try:
        if host == 'None':  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
