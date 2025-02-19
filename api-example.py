import io
import json
import wave

import requests
from io import BytesIO
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import read

# GET 请求示例
url = "http://127.0.0.1:9880/tts"
params = {
    "text": "先帝创业未半而中道崩殂",
    "text_lang": "zh",
    "prompt_lang": "ja",
    "ref_audio_path": r"E:\AI\GPT-SoVITS-v2-240821\output\slicer_opt\vocal_贝蒂纯净语音.mp3_10.wav_0000000000_0000153280.wav",
    "prompt_text": "ノックもしないで入り込んで、ずいぶんと無礼な奴なのよ.",
    "streaming_mode": "True",
    "media_type": "wav",
    "threshold": 20
}

# 流式获取音频
while True:
    params["text"] = input("请输入文本: ")
    response = requests.get(url, params=params, stream=True)

    if response.status_code == 200:
        print("正在接收流式音频...")
        sample_rate: int = int(response.headers.get('sample_rate'))

        # 使用 sounddevice.Stream 进行流式播放
        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype=np.int16) as stream:

            # 持续接收数据
            for chunk in response.iter_content(chunk_size=1024):  # 每次读取 1024 字节
                if chunk:
                    # 将二进制数据转换为 NumPy 数组
                    audio_chunk = np.frombuffer(chunk, dtype=np.int16)

                    # 立即播放当前音频块
                    stream.write(audio_chunk)
    elif response.status_code == 400:
        print(response.json())

#普通获取音频
# while True:
#     params["text"] = input("请输入文本: ")
#
#     response = requests.get(url, params=params)
#
#     if response.status_code == 200:
#         # 将二进制数据转换为音频流
#         audio_buffer = BytesIO(response.content)
#         sample_rate, audio_data = read(audio_buffer)
#
#         # 播放音频
#         sd.play(audio_data, sample_rate)
#         sd.wait()  # 等待播放完成
#     else:
#         print("错误:", response.json())
