import os
import uuid
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from main_receptionist import AIReceptionist

load_dotenv()

app = FastAPI(title="AI Digital Human Receptionist API")

# Setup directories
SESSIONS_DIR = Path("outputs/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the receptionist
receptionist = AIReceptionist()

# In-memory history (for demo purposes)
# In production, use a database or Redis
session_histories = {}

class ChatResponse(BaseModel):
    session_id: str
    user_text: str
    assistant_text: str
    audio_url: Optional[str] = None

class TextRequest(BaseModel):
    session_id: str
    text: str

@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(req: TextRequest):
    session_id = req.session_id
    history = session_histories.get(session_id, [])
    
    user_text, assistant_text = receptionist.handle_text_input(req.text, history)
    
    # Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    session_histories[session_id] = history[-10:] # Keep last 10 turns
    
    # Generate audio
    audio_filename = f"{uuid.uuid4()}.wav"
    audio_path = SESSIONS_DIR / audio_filename
    is_hindi = receptionist.lang_processor.is_hindi(assistant_text)
    receptionist.generate_speech(assistant_text, str(audio_path), is_hindi=is_hindi)
    
    return {
        "session_id": session_id,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_url": f"/audio/{audio_filename}"
    }

@app.post("/chat/audio", response_model=ChatResponse)
async def chat_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form("default")
):
    # Save uploaded file
    input_path = SESSIONS_DIR / f"input_{uuid.uuid4()}.wav"
    with open(input_path, "wb") as f:
        f.write(await audio_file.read())
    
    history = session_histories.get(session_id, [])
    
    user_text, assistant_text = receptionist.handle_voice_input(str(input_path), history)
    
    # Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    session_histories[session_id] = history[-10:]
    
    # Generate response audio
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = SESSIONS_DIR / output_filename
    is_hindi = receptionist.lang_processor.is_hindi(assistant_text)
    receptionist.generate_speech(assistant_text, str(output_path), is_hindi=is_hindi)
    
    return {
        "session_id": session_id,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_url": f"/audio/{output_filename}"
    }

# Serve audio files
from fastapi.responses import FileResponse

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = SESSIONS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(file_path))

@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    if session_id in session_histories:
        del session_histories[session_id]
    return {"status": "reset", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
