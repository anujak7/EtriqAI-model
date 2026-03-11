import argparse
import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="EtriqAI Hostel Receptionist")

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EtriqAI - Hostel Receptionist</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap");

    :root {
      --bg: #0f172a;
      --panel: rgba(30, 41, 59, 0.7);
      --text: #f8fafc;
      --accent: #38bdf8;
      --accent-2: #818cf8;
      --card-bg: rgba(51, 65, 85, 0.5);
      --border: rgba(148, 163, 184, 0.2);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Outfit", sans-serif;
      color: var(--text);
      background: linear-gradient(135deg, #0f172a, #1e293b, #030712);
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }

    .container {
      width: 90vw;
      height: 90vh;
      max-width: 1200px;
      background: var(--panel);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border);
      border-radius: 24px;
      display: grid;
      grid-template-columns: 350px 1fr;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      overflow: hidden;
    }

    .sidebar {
      background: rgba(15, 23, 42, 0.5);
      padding: 40px;
      display: flex;
      flex-direction: column;
      gap: 30px;
      border-right: 1px solid var(--border);
    }

    .logo {
      font-size: 2rem;
      font-weight: 700;
      background: linear-gradient(to right, var(--accent), var(--accent-2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 0;
    }

    .info-card {
      background: var(--card-bg);
      padding: 20px;
      border-radius: 16px;
      border: 1px solid var(--border);
    }

    .info-card h3 { margin: 0 0 10px; font-size: 0.9rem; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; }
    .info-card p { margin: 0; font-size: 1.1rem; font-weight: 600; }

    .main-chat {
      display: grid;
      grid-template-rows: 1fr auto;
      padding: 0;
      background: transparent;
    }

    .chat-box {
      padding: 40px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .message {
      max-width: 80%;
      padding: 16px 24px;
      border-radius: 20px;
      line-height: 1.5;
      font-size: 1.1rem;
      animation: fadeIn 0.3s ease;
    }

    .user-message {
      align-self: flex-end;
      background: var(--accent);
      color: #0f172a;
      border-bottom-right-radius: 4px;
    }

    .bot-message {
      align-self: flex-start;
      background: var(--card-bg);
      color: var(--text);
      border-bottom-left-radius: 4px;
      border: 1px solid var(--border);
    }

    .controls {
      padding: 40px;
      background: rgba(15, 23, 42, 0.8);
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .mic-button {
      width: 100%;
      padding: 20px;
      border-radius: 16px;
      border: none;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: #0f172a;
      font-size: 1.2rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.3s ease;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
    }

    .mic-button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px -10px var(--accent); }
    .mic-button:active { transform: translateY(0); }
    .mic-button.recording { background: #ef4444; animation: pulse 1.5s infinite; color: white; }

    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; display: inline-block; margin-right: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="sidebar">
      <h1 class="logo">EtriqAI</h1>
      <div class="info-card">
        <h3>Role</h3>
        <p>Hospital Receptionist</p>
      </div>
      <div class="info-card">
        <h3>Status</h3>
        <p><span class="status-dot"></span>Online</p>
      </div>
      <div class="info-card">
        <h3>Session</h3>
        <p id="session-id">Loading...</p>
      </div>
      <button class="mic-button" id="reset-btn" style="background: var(--card-bg); color: white; font-size: 0.9rem; padding: 10px;">New Session</button>
    </div>
    <div class="main-chat">
      <div class="chat-box" id="chat-box">
        <div class="message bot-message">Namaste! I am your Etriq Hospital Receptionist. How can I help you today? | नमस्ते! मैं आपका एट्रिक हॉस्पिटल रिसेप्शनिस्ट हूं। आज मैं आपकी क्या मदद कर सकता हूं?</div>
      </div>
      <div class="controls">
        <div id="status-text" style="color: var(--accent); font-size: 0.9rem; text-align: center;">Press the button and start talking</div>
        <button class="mic-button" id="mic-btn">
          <span id="mic-icon">🎤</span>
          <span id="mic-text">Start Talking</span>
        </button>
      </div>
    </div>
  </div>

  <script>
    let sessionId = "sess_" + Math.random().toString(36).substring(2, 9);
    document.getElementById('session-id').textContent = sessionId;

    const micBtn = document.getElementById('mic-btn');
    const chatBox = document.getElementById('chat-box');
    const statusText = document.getElementById('status-text');
    
    let recognition;
    if ('webkitSpeechRecognition' in window) {
      recognition = new webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'hi-IN';

      recognition.onstart = () => {
        micBtn.classList.add('recording');
        document.getElementById('mic-text').textContent = "Listening...";
        statusText.textContent = "Listening to your voice...";
      };

      recognition.onresult = async (event) => {
        const text = event.results[0][0].transcript;
        addMessage('user', text);
        await sendRequest(text);
      };

      recognition.onend = () => {
        micBtn.classList.remove('recording');
        document.getElementById('mic-text').textContent = "Start Talking";
        statusText.textContent = "Processing...";
      };

      recognition.onerror = (event) => {
        console.error("Speech Recognition Error", event.error);
        statusText.textContent = "Error: " + event.error;
      };
    }

    micBtn.addEventListener('click', () => {
      if (recognition) recognition.start();
    });

    document.getElementById('reset-btn').addEventListener('click', () => {
      sessionId = "sess_" + Math.random().toString(36).substring(2, 9);
      document.getElementById('session-id').textContent = sessionId;
      chatBox.innerHTML = '<div class="message bot-message">Session reset. How can I help you?</div>';
    });

    function addMessage(role, text) {
      const div = document.createElement('div');
      div.className = `message ${role}-message`;
      div.textContent = text;
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendRequest(text) {
      statusText.textContent = "Hospital Receptionist is thinking...";
      try {
        const response = await fetch('http://localhost:8000/chat/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, text: text })
        });
        const data = await response.json();
        addMessage('bot', data.assistant_text);
        
        if (data.audio_url) {
           const audio = new Audio(`http://localhost:8000${data.audio_url}`);
           audio.play();
        }
        statusText.textContent = "Ready";
      } catch (error) {
        console.error("Error:", error);
        statusText.textContent = "Connection Error";
        addMessage('bot', "I'm sorry, I couldn't connect to the server.");
      }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML_PAGE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
