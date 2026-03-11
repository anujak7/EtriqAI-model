import argparse
import io
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Avoid TensorFlow/Keras import path in transformers on this setup.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from voice_chat_whisper_coqui import (
    detect_emotion,
    generate_reply,
    is_xtts_model,
    load_emotion_pipe,
    load_llm,
    load_stt_pipe,
    load_tts,
    prepare_tts_text,
    synthesize_reply,
    transcribe,
    trim_history,
)


@dataclass
class RuntimeState:
    args: argparse.Namespace
    session_dir: Path
    tokenizer: Any
    model: Any
    stt_pipe: Any
    emotion_pipe: Any | None
    tts: Any | None
    speaker_wav: str | None
    history_by_session: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    turn_by_session: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class TextChatRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    text: str = Field(min_length=1)
    include_tts: bool = True
    tts_language: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastAPI server for Whisper STT + Qwen LoRA + XTTS + emotion detection."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="outputs/qwen2.5-1.5b-qlora-reception",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-small",
    )
    parser.add_argument(
        "--stt_language",
        type=str,
        default="hi",
    )
    parser.add_argument(
        "--coqui_model",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2",
    )
    parser.add_argument(
        "--tts_speaker",
        type=str,
        default="",
    )
    parser.add_argument(
        "--tts_language",
        type=str,
        default="hi",
    )
    parser.add_argument(
        "--speaker_wav",
        type=str,
        default="outputs/voice_chat_sessions/xtts_reference.wav",
        help="Reference wav for XTTS. If missing, upload via /speaker/reference.",
    )
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
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument(
        "--stt_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--tts_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--no_tts",
        action="store_true",
        help="Disable TTS in API responses.",
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
    )
    parser.add_argument(
        "--emotion_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--emotion_threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--no_emotion_detection",
        action="store_true",
    )
    parser.add_argument(
        "--coqui_tos_agreed",
        action="store_true",
        help="Set COQUI_TOS_AGREED=1 if you agree to CPML terms for XTTS.",
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        default="outputs/voice_chat_sessions",
    )
    return parser.parse_args()


def sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id).strip("_") or "default"


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}") from exc

    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype="float32"), int(sr)


def resolve_initial_speaker_wav(args: argparse.Namespace, xtts_mode: bool) -> str | None:
    if not xtts_mode:
        return None

    if args.speaker_wav:
        candidate = Path(args.speaker_wav)
        if candidate.exists():
            return str(candidate)

    fallback = Path(args.session_dir) / "xtts_reference.wav"
    if fallback.exists():
        return str(fallback)
    return None


def build_runtime(args: argparse.Namespace) -> RuntimeState:
    if args.coqui_tos_agreed:
        os.environ["COQUI_TOS_AGREED"] = "1"

    session_dir = Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LLM + adapter...")
    tokenizer, model = load_llm(args.base_model, args.adapter_dir)

    print("Loading Whisper STT...")
    stt_pipe = load_stt_pipe(args.whisper_model, args.stt_device)

    emotion_pipe = None
    if not args.no_emotion_detection:
        print("Loading Emotion Detector...")
        emotion_pipe = load_emotion_pipe(args.emotion_model, args.emotion_device)

    tts = None
    speaker_wav = None
    xtts_mode = is_xtts_model(args.coqui_model)
    if not args.no_tts:
        print("Loading Coqui TTS...")
        tts = load_tts(args.coqui_model, args.tts_device)
        speaker_wav = resolve_initial_speaker_wav(args, xtts_mode=xtts_mode)
        if xtts_mode and speaker_wav is None:
            print(
                "No XTTS reference wav found at startup. "
                "Upload one via POST /speaker/reference before include_tts=true."
            )

    return RuntimeState(
        args=args,
        session_dir=session_dir,
        tokenizer=tokenizer,
        model=model,
        stt_pipe=stt_pipe,
        emotion_pipe=emotion_pipe,
        tts=tts,
        speaker_wav=speaker_wav,
    )


app = FastAPI(title="Hospital Voice Chat API", version="1.0.0")
RUNTIME: RuntimeState | None = None


def get_runtime() -> RuntimeState:
    if RUNTIME is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized. Start server via voice_chat_api.py")
    return RUNTIME


def process_turn(
    runtime: RuntimeState,
    session_id: str,
    user_text: str,
    include_tts: bool,
    tts_language: str | None,
) -> dict[str, Any]:
    clean_session_id = sanitize_session_id(session_id)
    history = runtime.history_by_session.get(clean_session_id, [])

    detected_emotion, emotion_score = None, None
    if runtime.emotion_pipe is not None:
        detected_emotion, emotion_score = detect_emotion(
            emotion_pipe=runtime.emotion_pipe,
            text=user_text,
            threshold=runtime.args.emotion_threshold,
        )

    with runtime.lock:
        assistant_text = generate_reply(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            system_prompt=runtime.args.system_prompt,
            history=history,
            user_text=user_text,
            detected_emotion=detected_emotion,
            max_new_tokens=runtime.args.max_new_tokens,
            temperature=runtime.args.temperature,
            top_p=runtime.args.top_p,
        )

    history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    history = trim_history(history, runtime.args.history_turns)
    runtime.history_by_session[clean_session_id] = history

    tts_wav_path = None
    tts_text = None
    if include_tts and runtime.tts is not None:
        if is_xtts_model(runtime.args.coqui_model) and not runtime.speaker_wav:
            raise HTTPException(
                status_code=400,
                detail=(
                    "XTTS needs reference voice. Upload reference wav at /speaker/reference "
                    "or restart with --speaker_wav path."
                ),
            )

        turn_idx = runtime.turn_by_session.get(clean_session_id, 1)
        runtime.turn_by_session[clean_session_id] = turn_idx + 1

        use_tts_transliteration = (not runtime.args.no_tts_transliterate) and (
            not is_xtts_model(runtime.args.coqui_model)
        )
        tts_text = prepare_tts_text(assistant_text, transliterate=use_tts_transliteration)
        output_wav = runtime.session_dir / f"{clean_session_id}_turn_{turn_idx:04d}_assistant.wav"

        with runtime.lock:
            synthesize_reply(
                tts=runtime.tts,
                text=tts_text,
                output_wav=output_wav,
                tts_speaker=runtime.args.tts_speaker,
                tts_language=tts_language or runtime.args.tts_language,
                speaker_wav=runtime.speaker_wav,
            )

        tts_wav_path = str(output_wav)

    return {
        "session_id": clean_session_id,
        "user_text": user_text,
        "detected_emotion": detected_emotion,
        "emotion_score": emotion_score,
        "assistant_text": assistant_text,
        "tts_text": tts_text,
        "tts_wav_path": tts_wav_path,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    runtime = get_runtime()
    return {
        "status": "ok",
        "base_model": runtime.args.base_model,
        "adapter_dir": runtime.args.adapter_dir,
        "whisper_model": runtime.args.whisper_model,
        "emotion_model": None if runtime.args.no_emotion_detection else runtime.args.emotion_model,
        "coqui_model": None if runtime.args.no_tts else runtime.args.coqui_model,
        "speaker_wav": runtime.speaker_wav,
    }


@app.post("/chat/text")
def chat_text(req: TextChatRequest) -> dict[str, Any]:
    runtime = get_runtime()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    return process_turn(
        runtime=runtime,
        session_id=req.session_id,
        user_text=text,
        include_tts=req.include_tts,
        tts_language=req.tts_language,
    )


@app.post("/chat/audio")
async def chat_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form("default"),
    include_tts: bool = Form(True),
    tts_language: str | None = Form(None),
) -> dict[str, Any]:
    runtime = get_runtime()
    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    audio, sample_rate = decode_audio_bytes(content)
    with runtime.lock:
        user_text = transcribe(runtime.stt_pipe, audio, sample_rate, runtime.args.stt_language)

    if not user_text:
        raise HTTPException(status_code=400, detail="STT returned empty text.")

    result = process_turn(
        runtime=runtime,
        session_id=session_id,
        user_text=user_text,
        include_tts=include_tts,
        tts_language=tts_language,
    )
    result["transcription"] = user_text
    return result


@app.post("/speaker/reference")
async def upload_speaker_reference(audio_file: UploadFile = File(...)) -> dict[str, Any]:
    runtime = get_runtime()
    if runtime.tts is None or not is_xtts_model(runtime.args.coqui_model):
        raise HTTPException(status_code=400, detail="Speaker reference is only needed for XTTS mode.")

    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Validate audio file before saving.
    decode_audio_bytes(content)
    out_path = runtime.session_dir / "xtts_reference.wav"
    out_path.write_bytes(content)
    runtime.speaker_wav = str(out_path)
    return {"speaker_wav": str(out_path)}


@app.delete("/session/{session_id}")
def reset_session(session_id: str) -> dict[str, Any]:
    runtime = get_runtime()
    clean_session_id = sanitize_session_id(session_id)
    runtime.history_by_session.pop(clean_session_id, None)
    runtime.turn_by_session.pop(clean_session_id, None)
    return {"session_id": clean_session_id, "reset": True}


def main() -> None:
    global RUNTIME
    args = parse_args()
    RUNTIME = build_runtime(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
