from typing import List, Tuple, Optional
import numpy as np
import tqdm
import torch
from gpt_sovits.infer.inference import GPTSoVITSInference
from gpt_sovits.infer.text_utils import clean_and_cut_text
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import current_process


def worker_init(
    bert_path: str,
    cnhubert_base_path: str,
    gpt_path: str,
    sovits_path: str,
    prompt_text: Optional[str],
    prompt_language: str = "auto",
    prompt_audio_path: Optional[str] = None,
    prompt_audio_data: Optional[np.ndarray] = None,
    prompt_audio_sr: Optional[int] = None,
    device: Optional[str] = None,
    is_half: bool = True,
):
    global worker
    if device is None:
        if torch.cuda.is_available():
            cnt = torch.cuda.device_count()
            num = current_process()._identity[0]
            device = f"cuda:{num % cnt}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    worker = GPTSoVITSInference(
        bert_path=bert_path,
        cnhubert_base_path=cnhubert_base_path,
        device=device,
        is_half=is_half,
    )
    worker.load_gpt(gpt_path)
    worker.load_sovits(sovits_path)
    worker.set_prompt_audio(
        prompt_text,
        prompt_language,
        prompt_audio_path,
        prompt_audio_data,
        prompt_audio_sr,
    )


def worker_get_tts_wav_piece(
    text: str,
    text_language="auto",
    top_k=5,
    top_p=1,
    temperature=1,
):
    global worker
    return worker.get_tts_wav_piece(
        text,
        text_language,
        top_k,
        top_p,
        temperature,
    )


class GPTSoVITSInferencePool:
    pool: ProcessPoolExecutor

    def __init__(
        self,
        bert_path: str,
        cnhubert_base_path: str,
        gpt_path: str,
        sovits_path: str,
        prompt_text: Optional[str],
        prompt_language: str = "auto",
        prompt_audio_path: Optional[str] = None,
        prompt_audio_data: Optional[np.ndarray] = None,
        prompt_audio_sr: Optional[int] = None,
        device: Optional[str] = None,
        is_half: bool = True,
        max_workers: int = 4,
    ):
        self.pool = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(
                bert_path,
                cnhubert_base_path,
                gpt_path,
                sovits_path,
                prompt_text,
                prompt_language,
                prompt_audio_path,
                prompt_audio_data,
                prompt_audio_sr,
                device,
                is_half,
            ),
        )

    def get_tts_wav_stream(
        self,
        text: str,
        text_language="auto",
        top_k=5,
        top_p=1,
        temperature=1,
    ):
        tasks = clean_and_cut_text(text)
        futures = [
            self.pool.submit(
                worker_get_tts_wav_piece,
                task,
                text_language,
                top_k,
                top_p,
                temperature,
            )
            for task in tasks
        ]
        for future in futures:
            try:
                yield future.result()
            except Exception as e:
                self.pool.shutdown(wait=False)
                raise e

    def get_tts_wav(
        self,
        text: str,
        text_language="auto",
        top_k=5,
        top_p=1,
        temperature=1,
    ):
        audio_list: List[Tuple[int, np.ndarray]] = []
        total = len(clean_and_cut_text(text))
        for thing in tqdm.tqdm(
            self.get_tts_wav_stream(
                text,
                text_language,
                top_k,
                top_p,
                temperature,
            ),
            total=total,
        ):
            audio_list.append(thing)
        return audio_list[0][0], np.concatenate(
            [data for _, data in audio_list], axis=0
        )

    def __del__(self):
        self.pool.shutdown(wait=False)
