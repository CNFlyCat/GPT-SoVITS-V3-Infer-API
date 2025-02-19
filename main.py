import time

from gpt_sovits.infer import GPTSoVITSInference
import sounddevice as sd

inference = GPTSoVITSInference(
    bert_path="pretrained_models/chinese-roberta-wwm-ext-large",
    cnhubert_base_path="pretrained_models/chinese-hubert-base",
    is_half=True,  # Seems not support for the Colab environment
)

inference.load_sovits(r"E:\AI\GPT-SoVITS-v3-202502123fix2\SoVITS_weights_v3\BiCuiSi-v3_e2_s68.pth", version="v3")   # 设置版本
inference.load_gpt(r"E:\AI\GPT-SoVITS-v3-202502123fix2\GPT_weights_v3\BiCuiSi-v3-e15.ckpt")

# inference.load_sovits(r"E:\AI\GPT-SoVITS-v2-240821\SoVITS_weights_v2\BiCuiSi_e8_s272.pth", version="v2")   # 设置版本
# inference.load_gpt(r"E:\AI\GPT-SoVITS-v2-240821\GPT_weights_v2\BiCuiSi-e15.ckpt")

prompt_text = "ノックもしないで入り込んで、ずいぶんと無礼な奴なのよ。"
inference.set_prompt_audio(
    prompt_audio_path=r"E:\AI\GPT-SoVITS-v2-240821\output\slicer_opt\vocal_贝蒂纯净语音.mp3_10.wav_0000000000_0000153280.wav",
    prompt_language="ja",
    prompt_text=prompt_text,
)

while True:

    input_text = input("请输入文本: ")
    start_time = time.time()
    for sample_rate, audio_data in inference.get_tts_wav_stream(
            text_language="zh",
            text=input_text,
            sample_steps=16,
            version="v3",
            threshold=20
    ):
        if start_time > 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = 0
            print(f"音频生成耗时: {elapsed_time:.2f} 秒")
        sd.play(audio_data, sample_rate)
        sd.wait()

    # sample_rate, data = inference.get_tts_wav_stream(
    #     text_language="auto",
    #     text=input_text,
    #     sample_steps=8,
    #     version="v3"
    # )
    #
    # # 播放音频
    # sd.play(data, sample_rate)
