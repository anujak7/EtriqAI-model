import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Avoid TensorFlow/Keras import path in transformers on this setup.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from peft import PeftModel
from TTS.api import TTS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime voice chat using Whisper STT + Qwen LoRA adapter + Coqui TTS."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF base model used for LoRA training.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/qwen2.5-1.5b-qlora-reception",
        help="Path to trained LoRA adapter directory.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-small",
        help="Whisper model id for speech-to-text.",
    )
    parser.add_argument(
        "--stt_language",
        type=str,
        default="hi",
        help="Whisper transcription language code, e.g. hi/en. Use auto to disable forcing.",
    )
    parser.add_argument(
        "--coqui_model",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model name.",
    )
    parser.add_argument(
        "--tts_speaker",
        type=str,
        default="",
        help="Speaker id for multi-speaker Coqui models.",
    )
    parser.add_argument(
        "--tts_language",
        type=str,
        default="hi",
        help="Language code for multi-lingual Coqui models.",
    )
    parser.add_argument(
        "--speaker_wav",
        type=str,
        default=None,
        help="Optional voice reference wav for models like xtts.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--record_seconds", type=float, default=7.0)
    parser.add_argument(
        "--reference_seconds",
        type=float,
        default=8.0,
        help="Seconds to record for XTTS reference voice when speaker_wav is not provided.",
    )
    parser.add_argument(
        "--no_auto_reference",
        action="store_true",
        help="Disable automatic reference voice recording for XTTS when speaker_wav is missing.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a helpful hospital reception assistant. "
            "Give accurate, practical, and concise answers with clear next steps. "
            "If details are uncertain, say so and suggest front-desk confirmation. "
            "Reply in the user's language."
        ),
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        default="outputs/voice_chat_sessions",
        help="Directory to store recorded and generated audio files.",
    )
    parser.add_argument(
        "--stt_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Whisper device. Keep cpu on low-VRAM GPUs.",
    )
    parser.add_argument(
        "--tts_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Coqui TTS device. Keep cpu on low-VRAM GPUs.",
    )
    parser.add_argument(
        "--no_playback",
        action="store_true",
        help="Do not play generated TTS audio. Only save wav files.",
    )
    parser.add_argument(
        "--no_tts_transliterate",
        action="store_true",
        help="Disable automatic non-ASCII to ASCII transliteration before TTS.",
    )
    parser.add_argument(
        "--emotion_model",
        type=str,
        default="j-hartmann/emotion-english-distilroberta-base",
        help="HF text-classification model id for emotion detection.",
    )
    parser.add_argument(
        "--emotion_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Emotion model device. Keep cpu on low-VRAM GPUs.",
    )
    parser.add_argument(
        "--emotion_threshold",
        type=float,
        default=0.35,
        help="Minimum confidence to apply detected emotion to reply style.",
    )
    parser.add_argument(
        "--no_emotion_detection",
        action="store_true",
        help="Disable emotion detection step before response generation.",
    )
    parser.add_argument(
        "--coqui_tos_agreed",
        action="store_true",
        help=(
            "Set COQUI_TOS_AGREED=1 for XTTS CPML consent to avoid interactive prompt. "
            "Use only if you agree to CPML terms."
        ),
    )
    return parser.parse_args()


def configure_console_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ensure_paths(args: argparse.Namespace) -> Path:
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    session_dir = Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def load_llm(base_model: str, adapter_dir: str) -> tuple[Any, Any]:
    hf_token = os.getenv("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=hf_token)
    except Exception as exc:
        print(
            "Tokenizer load from adapter failed; falling back to base model tokenizer. "
            f"Reason: {exc}"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    if not getattr(tokenizer, "chat_template", None):
        print("Adapter tokenizer has no chat template; using base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    base.config.use_cache = True

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def load_stt_pipe(model_id: str, stt_device: str) -> Any:
    if stt_device == "cuda" and torch.cuda.is_available():
        device = 0
        dtype = torch.float16
    else:
        device = -1
        dtype = torch.float32

    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            framework="pt",
            device=device,
            torch_dtype=dtype,
        )
    except TypeError:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            framework="pt",
            device=device,
            dtype=dtype,
        )


def load_emotion_pipe(model_id: str, emotion_device: str) -> Any:
    if emotion_device == "cuda" and torch.cuda.is_available():
        device = 0
        dtype = torch.float16
    else:
        device = -1
        dtype = torch.float32

    try:
        return pipeline(
            task="text-classification",
            model=model_id,
            framework="pt",
            device=device,
            torch_dtype=dtype,
        )
    except TypeError:
        return pipeline(
            task="text-classification",
            model=model_id,
            framework="pt",
            device=device,
            dtype=dtype,
        )


def load_tts(model_name: str, tts_device: str) -> TTS:
    use_gpu = tts_device == "cuda" and torch.cuda.is_available()
    restore_torch_load = None

    # Torch 2.6 changed torch.load(weights_only=True by default), while XTTS
    # checkpoints require full object deserialization during model init.
    if is_xtts_model(model_name):
        original_torch_load = torch.load

        def _torch_load_compat(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat
        restore_torch_load = original_torch_load

        # XTTS expects GPT2InferenceModel to expose generate(), but newer
        # Transformers versions moved this behavior behind GenerationMixin.
        try:
            from transformers.generation.utils import GenerationMixin
            from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel

            for name in dir(GenerationMixin):
                if name.startswith("__"):
                    continue
                if hasattr(GPT2InferenceModel, name):
                    continue
                setattr(GPT2InferenceModel, name, getattr(GenerationMixin, name))
        except Exception as exc:
            print(f"Warning: XTTS generation compatibility patch failed: {exc}")

    try:
        return TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)
    finally:
        if restore_torch_load is not None:
            torch.load = restore_torch_load


def is_xtts_model(model_name: str) -> bool:
    return "xtts" in model_name.lower()


def resolve_speaker_wav(
    args: argparse.Namespace, session_dir: Path, needs_reference: bool
) -> str | None:
    if args.speaker_wav:
        speaker_path = Path(args.speaker_wav)
        if not speaker_path.exists():
            raise FileNotFoundError(f"speaker_wav not found: {speaker_path}")
        return str(speaker_path)

    if not needs_reference:
        return None

    if args.no_auto_reference:
        raise ValueError(
            "XTTS model needs reference voice. Pass --speaker_wav or remove --no_auto_reference."
        )

    print(
        "\nXTTS reference voice needed. "
        f"Press Enter and speak naturally for {args.reference_seconds:.1f}s."
    )
    input("Press Enter to record reference > ")
    ref_audio = record_from_mic(args.sample_rate, args.reference_seconds)
    ref_wav = session_dir / "xtts_reference.wav"
    sf.write(ref_wav, ref_audio, args.sample_rate)
    print(f"Reference voice saved: {ref_wav}\n")
    return str(ref_wav)


def trim_history(history: list[dict[str, str]], turns: int) -> list[dict[str, str]]:
    if turns <= 0:
        return []
    return history[-(turns * 2) :]


def record_from_mic(sample_rate: int, seconds: float) -> np.ndarray:
    total_samples = int(sample_rate * seconds)
    print(f"Recording for {seconds:.1f}s...")
    data = sd.rec(total_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(data)


def transcribe(
    stt_pipe: Any, audio: np.ndarray, sample_rate: int, language: str
) -> str:
    gen_kwargs: dict[str, Any] = {"task": "transcribe"}
    if language and language.lower() != "auto":
        gen_kwargs["language"] = language

    result = stt_pipe(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs=gen_kwargs,
    )
    text = result["text"].strip()
    return text


def generate_reply(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    history: list[dict[str, str]],
    user_text: str,
    detected_emotion: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if detected_emotion:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Detected user emotion: {detected_emotion}. "
                    "Respond with appropriate empathy and calm tone."
                ),
            }
        )
    messages = messages + history + [
        {"role": "user", "content": user_text}
    ]

    model_input = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(model_input, dict):
        model_input = model_input["input_ids"]
    elif hasattr(model_input, "input_ids"):
        model_input = model_input.input_ids

    llm_device = next(model.parameters()).device
    model_input = model_input.to(llm_device)
    attention_mask = torch.ones_like(model_input)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=model_input,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    generated_ids = output_ids[0, model_input.shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


def synthesize_reply(
    tts: TTS,
    text: str,
    output_wav: Path,
    tts_speaker: str,
    tts_language: str,
    speaker_wav: str | None,
) -> None:
    kwargs: dict[str, Any] = {"text": text, "file_path": str(output_wav)}

    if speaker_wav:
        kwargs["speaker_wav"] = speaker_wav

    if getattr(tts, "is_multi_speaker", False) and tts_speaker:
        kwargs["speaker"] = tts_speaker

    if getattr(tts, "is_multi_lingual", False) and tts_language:
        kwargs["language"] = tts_language

    tts.tts_to_file(**kwargs)


def prepare_tts_text(text: str, transliterate: bool) -> str:
    if not transliterate:
        return text
    if text.isascii():
        return text

    try:
        from unidecode import unidecode

        normalized = unidecode(text)
        normalized = " ".join(normalized.split())
        return normalized if normalized else text
    except Exception:
        return text


def play_wav(path: Path) -> None:
    audio, sr = sf.read(path, dtype="float32")
    sd.play(audio, sr)
    sd.wait()


def detect_emotion(
    emotion_pipe: Any, text: str, threshold: float
) -> tuple[str | None, float | None]:
    try:
        result = emotion_pipe(text, top_k=None, truncation=True)
    except TypeError:
        # Compatibility with older transformers where top_k may differ.
        result = emotion_pipe(text)
    except Exception:
        return None, None

    if not result:
        return None, None

    if isinstance(result, list) and result and isinstance(result[0], list):
        candidates = result[0]
    elif isinstance(result, list):
        candidates = result
    elif isinstance(result, dict):
        candidates = [result]
    else:
        return None, None

    best = None
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if "label" not in item or "score" not in item:
            continue
        if best is None or float(item["score"]) > float(best["score"]):
            best = item

    if best is None:
        return None, None

    score = float(best["score"])
    if score < threshold:
        return None, score

    label = str(best["label"]).strip().lower()
    return label, score


def main() -> None:
    configure_console_utf8()
    args = parse_args()
    session_dir = ensure_paths(args)
    xtts_mode = is_xtts_model(args.coqui_model)

    if xtts_mode and args.coqui_tos_agreed:
        os.environ["COQUI_TOS_AGREED"] = "1"

    print("Loading LLM + adapter...")
    tokenizer, model = load_llm(args.base_model, args.adapter_dir)

    print("Loading Whisper STT...")
    stt_pipe = load_stt_pipe(args.whisper_model, args.stt_device)

    emotion_pipe = None
    if not args.no_emotion_detection:
        print("Loading Emotion Detector...")
        emotion_pipe = load_emotion_pipe(args.emotion_model, args.emotion_device)

    print("Loading Coqui TTS...")
    try:
        tts = load_tts(args.coqui_model, args.tts_device)
    except Exception as exc:
        if "terms of service" in str(exc).lower():
            raise RuntimeError(
                "XTTS requires CPML consent. Re-run with --coqui_tos_agreed "
                "or set environment variable COQUI_TOS_AGREED=1."
            ) from exc
        raise

    speaker_wav_for_tts = resolve_speaker_wav(
        args=args, session_dir=session_dir, needs_reference=xtts_mode
    )

    print("\nVoice chat ready.")
    print("Controls: press Enter to record, type `t` for text input, `q` to quit.\n")

    history: list[dict[str, str]] = []
    turn_idx = 1

    while True:
        mode = input("[Enter/t/q] > ").strip().lower()
        if mode == "q":
            break

        if mode == "t":
            user_text = input("You (text): ").strip()
        elif mode == "":
            audio = record_from_mic(args.sample_rate, args.record_seconds)
            input_wav = session_dir / f"turn_{turn_idx:03d}_user.wav"
            sf.write(input_wav, audio, args.sample_rate)

            user_text = transcribe(stt_pipe, audio, args.sample_rate, args.stt_language)
            print(f"You (STT): {user_text}")
        else:
            print("Invalid input. Use Enter for mic, `t` for text, `q` to quit.")
            continue

        if not user_text:
            print("Empty input detected. Try again.")
            continue

        detected_emotion, emotion_score = None, None
        if emotion_pipe is not None:
            detected_emotion, emotion_score = detect_emotion(
                emotion_pipe=emotion_pipe,
                text=user_text,
                threshold=args.emotion_threshold,
            )
            if detected_emotion and emotion_score is not None:
                print(f"Detected emotion: {detected_emotion} ({emotion_score:.2f})")

        assistant_text = generate_reply(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            history=history,
            user_text=user_text,
            detected_emotion=detected_emotion,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Assistant: {assistant_text}")
        use_tts_transliteration = (not args.no_tts_transliterate) and (not xtts_mode)
        tts_text = prepare_tts_text(
            assistant_text, transliterate=use_tts_transliteration
        )
        if tts_text != assistant_text:
            print(f"TTS text (normalized): {tts_text}")

        output_wav = session_dir / f"turn_{turn_idx:03d}_assistant.wav"
        synthesize_reply(
            tts=tts,
            text=tts_text,
            output_wav=output_wav,
            tts_speaker=args.tts_speaker,
            tts_language=args.tts_language,
            speaker_wav=speaker_wav_for_tts,
        )

        print(f"TTS saved: {output_wav}")
        if not args.no_playback:
            play_wav(output_wav)

        history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        history = trim_history(history, args.history_turns)
        turn_idx += 1

    print("Voice chat ended.")


if __name__ == "__main__":
    main()
