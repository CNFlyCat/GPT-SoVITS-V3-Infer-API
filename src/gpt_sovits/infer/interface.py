from gpt_sovits.infer.inference import GPTSoVITSInference
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Union, TYPE_CHECKING
from pathlib import Path
from threading import Lock
import os, sys

if TYPE_CHECKING:
    import numpy as np


class ConfigData(BaseModel):
    bert_path: Path
    cnhubert_base_path: Path
    models: Dict[str, Tuple[Path, Path]]
    prompts: Dict[str, Tuple[Path, Path]]


def read_config_data(config_data_base: str) -> ConfigData:
    config_data_base = Path(config_data_base)
    model_files = os.listdir(config_data_base / "models")
    models = {
        model_file.split(".")[0]: (
            config_data_base / "models" / model_file,
            config_data_base / "models" / model_file.replace(".pth", ".ckpt"),
        )
        for model_file in model_files
        if model_file.endswith(".pth")
        and model_file.replace(".pth", ".ckpt") in model_files
    }
    prompt_files = os.listdir(Path(config_data_base) / "prompts")
    prompts = {
        prompt_file.split(".")[0]: (
            config_data_base / "prompts" / prompt_file,
            config_data_base / "prompts" / prompt_file.replace(".txt", ".wav"),
        )
        for prompt_file in prompt_files
        if prompt_file.endswith(".txt")
        and prompt_file.replace(".txt", ".wav") in prompt_files
    }
    cnhubert_base_path = str(Path(config_data_base) / "chinese-hubert-base")
    bert_path = str(Path(config_data_base) / "chinese-roberta-wwm-ext-large")
    return ConfigData(
        bert_path=bert_path,
        cnhubert_base_path=cnhubert_base_path,
        models=models,
        prompts=prompts,
    )


PromptType = Union[str, Tuple[str, int, "np.ndarray"]]


class GPTSoVITSInterfaceSimple:
    config_data: ConfigData
    working_model: Optional[str]
    working_prompt: Optional[str]

    inference_worker: GPTSoVITSInference
    inference_worker_lock: Lock

    def __init__(
        self,
        config_data_base: str,
        device: Optional[str] = None,
        is_half: Optional[bool] = True,
    ):
        self.config_data = read_config_data(config_data_base)
        self.working_model = None
        self.working_prompt = None
        self.inference_worker = GPTSoVITSInference(
            self.config_data.bert_path,
            self.config_data.cnhubert_base_path,
            device=device,
            is_half=is_half,
        )
        self.inference_worker_lock = Lock()

    def _load_model(self, model_name: str):
        self.inference_worker.load_sovits(
            self.config_data.models[model_name][0],
        )
        self.inference_worker.load_gpt(
            self.config_data.models[model_name][1],
        )

    def _load_prompt(self, prompt_name: str):
        with open(self.config_data.prompts[prompt_name][0], "r") as f:
            prompt_text = f.read().strip()
        self.inference_worker.set_prompt_audio(
            prompt_text=prompt_text,
            prompt_audio_path=self.config_data.prompts[prompt_name][1],
        )

    def _load_things(self, model_name: str, prompt: PromptType):
        if model_name != self.working_model:
            self._load_model(model_name)
            self.working_model = model_name
        if isinstance(prompt, str):
            if prompt != self.working_prompt:
                self._load_prompt(prompt)
                self.working_prompt = prompt
        else:
            self.inference_worker.set_prompt_audio(
                prompt_text=prompt[0],
                prompt_audio_data=prompt[2],
                prompt_audio_sr=prompt[1],
            )
            self.working_prompt = None

    def generate(
        self,
        model_name: str,
        prompt: PromptType,
        text: str,
        text_language="auto",
        top_k=5,
        top_p=1,
        temperature=1,
    ):
        with self.inference_worker_lock:
            self._load_things(model_name, prompt)
            return self.inference_worker.get_tts_wav(
                text=text,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

    def generate_stream(
        self,
        model_name: str,
        prompt: PromptType,
        text: str,
        text_language="auto",
        top_k=5,
        top_p=1,
        temperature=1,
    ):
        with self.inference_worker_lock:
            self._load_things(model_name, prompt)
            for thing in self.inference_worker.get_tts_wav_stream(
                text=text,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                yield thing

        return None
