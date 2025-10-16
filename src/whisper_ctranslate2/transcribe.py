import inspect
import os
import sys
import warnings

from typing import BinaryIO, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import tqdm

from faster_whisper import BatchedInferencePipeline, WhisperModel

from .languages import LANGUAGES
from .writers import format_timestamp

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        return string


class TranscriptionOptions(NamedTuple):
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperature: List[float]
    initial_prompt: Optional[str]
    prefix: Optional[str]
    hotwords: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    #    max_initial_timestamp: float
    word_timestamps: bool
    print_colors: bool
    prepend_punctuations: str
    append_punctuations: str
    hallucination_silence_threshold: Optional[float]
    vad_filter: bool
    vad_threshold: Optional[float]
    vad_min_speech_duration_ms: Optional[int]
    vad_max_speech_duration_s: Optional[int]
    vad_min_silence_duration_ms: Optional[int]
    multilingual: bool


class _MPSWord(NamedTuple):
    start: float
    end: float
    word: str
    probability: float


class _MPSSegment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    temperature: Optional[float]
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[_MPSWord]]


class _MPSTranscriptionInfo(NamedTuple):
    language: Optional[str]
    language_probability: Optional[float]
    duration: float


class _MPSWhisperModel:
    _UNSUPPORTED_OPTION_WARNINGS: Dict[str, bool] = {}

    def __init__(
        self,
        model_path: str,
        threads: int,
        cache_directory: Optional[str],
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            import whisper
            from whisper.decoding import DecodingOptions
            from whisper import _downloads
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Using the MPS backend requires the optional 'openai-whisper' and "
                "'torch' dependencies. Install them with `pip install openai-whisper`."
            ) from exc

        if not torch.backends.mps.is_available():  # pragma: no cover - hardware specific
            raise RuntimeError(
                "Metal (MPS) backend requested but torch reports that MPS is not "
                "available. Ensure you are running on Apple Silicon with macOS 13 or "
                "newer and PyTorch built with MPS support."
            )

        if model_path and os.path.isdir(model_path):
            converted_model = os.path.join(model_path, "model.bin")
            if os.path.exists(converted_model):
                raise RuntimeError(
                    "CTranslate2 converted models are not compatible with the MPS "
                    "backend. Provide an original Whisper model name (e.g. 'small', "
                    "'medium.en') instead of a CTranslate2 directory."
                )

        self._torch = torch
        self._whisper = whisper
        self._downloads = _downloads
        self._transcribe_params = set(inspect.signature(whisper.transcribe).parameters)
        self._decode_option_fields = set(DecodingOptions.__annotations__.keys())

        if threads and threads > 0:
            torch.set_num_threads(threads)

        download_root = cache_directory
        if local_files_only:
            download_root = download_root or _downloads.default_download_root()
            if not self._model_files_present(model_path, download_root):
                raise RuntimeError(
                    "Requested --local_files_only but the Whisper model weights were "
                    "not found locally. Download the model once without "
                    "--local_files_only before using MPS offline."
                )

        self._model = whisper.load_model(
            model_path, device="mps", download_root=cache_directory
        )

    def _model_files_present(self, model_name: str, download_root: str) -> bool:
        if os.path.isfile(model_name):
            return True
        candidate_paths = [
            os.path.join(download_root, model_name + ".pt"),
            os.path.join(download_root, model_name, "model.pt"),
        ]
        return any(os.path.exists(path) for path in candidate_paths)

    @classmethod
    def _warn_once(cls, option_name: str, message: str) -> None:
        if not cls._UNSUPPORTED_OPTION_WARNINGS.get(option_name):
            warnings.warn(message)
            cls._UNSUPPORTED_OPTION_WARNINGS[option_name] = True

    def _prepare_transcribe_kwargs(
        self,
        options: "TranscriptionOptions",
        task: str,
        language: Optional[str],
        word_timestamps: bool,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        temperature_schedule = list(options.temperature)
        base_temperature = temperature_schedule[0] if temperature_schedule else 0.0
        temperature_increment = None
        if len(temperature_schedule) > 1:
            temperature_increment = temperature_schedule[1] - temperature_schedule[0]

        decode_candidates: Dict[str, object] = {
            "task": task,
            "language": language,
            "beam_size": options.beam_size,
            "best_of": options.best_of,
            "patience": options.patience,
            "length_penalty": options.length_penalty,
            "repetition_penalty": options.repetition_penalty,
            "fp16": False,
        }

        if options.no_repeat_ngram_size:
            decode_candidates["no_repeat_ngram_size"] = options.no_repeat_ngram_size

        decode_options = {
            key: value
            for key, value in decode_candidates.items()
            if key in self._decode_option_fields and value is not None
        }

        transcribe_candidates: Dict[str, object] = {
            "temperature": base_temperature,
            "temperature_increment_on_fallback": temperature_increment,
            "compression_ratio_threshold": options.compression_ratio_threshold,
            "logprob_threshold": options.log_prob_threshold,
            "no_speech_threshold": options.no_speech_threshold,
            "condition_on_previous_text": options.condition_on_previous_text,
            "initial_prompt": options.initial_prompt,
            "prefix": options.prefix,
            "suppress_blank": options.suppress_blank,
            "suppress_tokens": options.suppress_tokens if options.suppress_tokens else None,
            "word_timestamps": word_timestamps,
            "prepend_punctuations": options.prepend_punctuations,
            "append_punctuations": options.append_punctuations,
            "vad_filter": options.vad_filter,
        }

        transcribe_kwargs = {
            key: value
            for key, value in transcribe_candidates.items()
            if key in self._transcribe_params and value is not None
        }

        if "temperature_increment_on_fallback" not in self._transcribe_params:
            transcribe_kwargs.pop("temperature_increment_on_fallback", None)

        if "vad_filter" not in self._transcribe_params:
            transcribe_kwargs.pop("vad_filter", None)

        return transcribe_kwargs, decode_options

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        *,
        task: str,
        language: Optional[str],
        options: "TranscriptionOptions",
        verbose: bool,
        print_colors: bool,
    ) -> Tuple[List[_MPSSegment], _MPSTranscriptionInfo]:
        unsupported = []
        if options.hotwords:
            unsupported.append("hotwords")
        if options.hallucination_silence_threshold:
            unsupported.append("hallucination_silence_threshold")
        for option in unsupported:
            self._warn_once(
                option,
                f"Option '{option}' is not supported by the MPS backend and will be ignored.",
            )

        word_timestamps = True if print_colors else options.word_timestamps

        transcribe_kwargs, decode_options = self._prepare_transcribe_kwargs(
            options, task, language, word_timestamps
        )

        result = self._model.transcribe(
            audio,
            verbose=verbose,
            **transcribe_kwargs,
            **decode_options,
        )

        segments: List[_MPSSegment] = []
        for segment in result.get("segments", []):
            words = segment.get("words") or None
            if words:
                words = [
                    _MPSWord(
                        start=word.get("start", 0.0),
                        end=word.get("end", 0.0),
                        word=word.get("word", ""),
                        probability=word.get("probability", 0.0),
                    )
                    for word in words
                ]

            segments.append(
                _MPSSegment(
                    id=segment.get("id", len(segments)),
                    seek=segment.get("seek", 0),
                    start=segment.get("start", 0.0),
                    end=segment.get("end", 0.0),
                    text=segment.get("text", ""),
                    tokens=segment.get("tokens", []),
                    avg_logprob=segment.get("avg_logprob", 0.0),
                    temperature=segment.get("temperature"),
                    compression_ratio=segment.get("compression_ratio", 0.0),
                    no_speech_prob=segment.get("no_speech_prob", 0.0),
                    words=words,
                )
            )

        duration = result.get("duration")
        if duration is None and segments:
            duration = segments[-1].end
        duration = float(duration or 0.0)

        info = _MPSTranscriptionInfo(
            language=result.get("language"),
            language_probability=result.get("language_probability"),
            duration=duration,
        )

        return segments, info


class Transcribe:
    @staticmethod
    def _resolve_device_and_compute_type(device: str, compute_type: str):
        return device, compute_type

    def _get_colored_text(self, words):
        k_colors = [
            "\033[38;5;196m",
            "\033[38;5;202m",
            "\033[38;5;208m",
            "\033[38;5;214m",
            "\033[38;5;220m",
            "\033[38;5;226m",
            "\033[38;5;190m",
            "\033[38;5;154m",
            "\033[38;5;118m",
            "\033[38;5;82m",
        ]

        text_words = ""

        n_colors = len(k_colors)
        for word in words:
            p = word.probability
            col = max(0, min(n_colors - 1, (int)(pow(p, 3) * n_colors)))
            end_mark = "\033[0m"
            text_words += f"{k_colors[col]}{word.word}{end_mark}"

        return text_words

    def _get_vad_parameters_dictionary(self, options):
        vad_parameters = {}

        if options.vad_threshold:
            vad_parameters["threshold"] = options.vad_threshold

        if options.vad_min_speech_duration_ms:
            vad_parameters["min_speech_duration_ms"] = (
                options.vad_min_speech_duration_ms
            )

        if options.vad_max_speech_duration_s:
            vad_parameters["max_speech_duration_s"] = options.vad_max_speech_duration_s

        if options.vad_min_silence_duration_ms:
            vad_parameters["min_silence_duration_ms"] = (
                options.vad_min_silence_duration_ms
            )

        return vad_parameters

    def __init__(
        self,
        model_path: str,
        device: str,
        device_index: Union[int, List[int]],
        compute_type: str,
        threads: int,
        cache_directory: str,
        local_files_only: bool,
        batched: bool,
        batch_size: int = None,
    ):
        self._is_mps = device in {"mps", "metal"}

        if self._is_mps:
            if device == "metal":
                warnings.warn("Device 'metal' is deprecated; use 'mps' instead.")

            if compute_type not in {"auto", "default"}:
                warnings.warn(
                    "MPS backend ignores --compute_type. Using float32 precision."
                )

            if device_index not in {0, None}:
                warnings.warn("MPS backend only supports device index 0; ignoring value")

            if batched:
                warnings.warn(
                    "Batched inference is not supported on the MPS backend; running "
                    "without batching."
                )

            self.model = _MPSWhisperModel(
                model_path=model_path,
                threads=threads,
                cache_directory=cache_directory,
                local_files_only=local_files_only,
            )
            self.batched_model = None
            self.batch_size = None
            return

        normalized_device, normalized_compute_type = self._resolve_device_and_compute_type(
            device, compute_type
        )

        self.model = WhisperModel(
            model_path,
            device=normalized_device,
            device_index=device_index,
            compute_type=normalized_compute_type,
            cpu_threads=threads,
            download_root=cache_directory,
            local_files_only=local_files_only,
        )

        self.batch_size = batch_size
        if batched:
            self.batched_model = BatchedInferencePipeline(model=self.model)
        else:
            self.batched_model = None

    def inference(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        task: str,
        language: str,
        verbose: bool,
        live: bool,
        options: TranscriptionOptions,
    ):
        vad_parameters = self._get_vad_parameters_dictionary(options)

        if self.batched_model:
            model = self.batched_model
            vad = True
        else:
            model = self.model
            vad = options.vad_filter

        if isinstance(model, _MPSWhisperModel):
            segments, info = model.transcribe(
                audio=audio,
                task=task,
                language=language,
                options=options,
                verbose=verbose,
                print_colors=options.print_colors,
            )
        else:
            batch_size = (
                {"batch_size": self.batch_size} if self.batch_size is not None else {}
            )

            segments, info = model.transcribe(
                audio=audio,
                language=language,
                task=task,
                beam_size=options.beam_size,
                best_of=options.best_of,
                patience=options.patience,
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                temperature=options.temperature,
                compression_ratio_threshold=options.compression_ratio_threshold,
                log_prob_threshold=options.log_prob_threshold,
                no_speech_threshold=options.no_speech_threshold,
                condition_on_previous_text=options.condition_on_previous_text,
                prompt_reset_on_temperature=options.prompt_reset_on_temperature,
                initial_prompt=options.initial_prompt,
                prefix=options.prefix,
                hotwords=options.hotwords,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                word_timestamps=True if options.print_colors else options.word_timestamps,
                prepend_punctuations=options.prepend_punctuations,
                append_punctuations=options.append_punctuations,
                hallucination_silence_threshold=options.hallucination_silence_threshold,
                vad_filter=vad,
                vad_parameters=vad_parameters,
                **batch_size,
                multilingual=options.multilingual,
            )

        language_name = LANGUAGES[info.language].title()
        if not live:
            print(
                "Detected language '%s' with probability %f"
                % (language_name, info.language_probability)
            )

        list_segments = []
        last_pos = 0
        accumulated_inc = 0
        all_text = ""
        with tqdm.tqdm(
            total=info.duration, unit="seconds", disable=verbose or live is not False
        ) as pbar:
            for segment in segments:
                start, end, text = segment.start, segment.end, segment.text
                all_text += segment.text

                if verbose or options.print_colors:
                    if options.print_colors and segment.words:
                        text = self._get_colored_text(segment.words)
                    else:
                        text = segment.text

                    if not live:
                        line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                        print(make_safe(line))

                segment_dict = segment._asdict()
                if segment.words:
                    segment_dict["words"] = [word._asdict() for word in segment.words]

                list_segments.append(segment_dict)
                duration = segment.end - last_pos
                increment = (
                    duration
                    if accumulated_inc + duration < info.duration
                    else info.duration - accumulated_inc
                )
                accumulated_inc += increment
                last_pos = segment.end
                pbar.update(increment)

        return dict(
            text=all_text,
            segments=list_segments,
            language=info.language,
        )
