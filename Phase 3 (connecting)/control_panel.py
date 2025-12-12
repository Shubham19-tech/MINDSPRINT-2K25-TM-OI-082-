from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import uuid, base64, tempfile, subprocess, json, requests
from datetime import datetime
from pathlib import Path

# ---------------- FASTAPI APP ---------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def serve_page():
    return FileResponse("index.html")


# ---------------- CONFIG ---------------- #

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"   # FIXED URL
LLM_MODEL = "llama2:7b"                               # MATCHES YOUR WORKING MODEL

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# Whisper STT
try:
    from faster_whisper import WhisperModel
    WHISPER = WhisperModel("base", device="cpu")
except:
    WHISPER = None

# Text-To-Speech
try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
except:
    TTS_ENGINE = None


# ---------------- HELPERS ---------------- #

def convert_to_wav(webm_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def transcribe_audio(path):
    if WHISPER is None:
        return "(Whisper not installed)"
    segments, _ = WHISPER.transcribe(path)
    return " ".join([seg.text for seg in segments])


def ask_llama(prompt: str):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })
        return r.json().get("response", "(Empty response)")
    except Exception as e:
        return f"Error contacting LLaMA: {e}"


def make_tts(text, outfile):
    if TTS_ENGINE is None:
        return None
    TTS_ENGINE.setProperty("rate", 160)
    TTS_ENGINE.save_to_file(text, outfile)
    TTS_ENGINE.runAndWait()
    return outfile


# ---------------- LLaMA TEST ENDPOINT ---------------- #

@app.get("/test-llama")
def test_llama():
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": "Say OK",
            "stream": False
        })
        return {"response": r.json().get("response")}
    except Exception as e:
        return {"error": str(e)}


# ---------------- WEBSOCKET INTERVIEW ---------------- #

@app.websocket("/ws/interview")
async def interview(ws: WebSocket):
    await ws.accept()

    session_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / session_id
    temp_dir.mkdir(exist_ok=True)

    chunks = []
    session_log = {
        "session_id": session_id,
        "started": str(datetime.now()),
        "exchanges": []
    }

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            # ------------- AUDIO CHUNK ------------- #
            if mtype == "audio_chunk":
                raw = base64.b64decode(msg["data"])
                p = temp_dir / f"{uuid.uuid4().hex}.webm"
                p.write_bytes(raw)
                chunks.append(str(p))

                await ws.send_json({"type": "ack"})

            # ------------- FINALIZE ANSWER --------- #
            elif mtype == "finalize":

                merged = temp_dir / "merged.webm"
                wav = temp_dir / "audio.wav"

                # Merge chunks
                listfile = temp_dir / "list.txt"
                with open(listfile, "w") as f:
                    for c in chunks:
                        f.write(f"file '{c}'\n")

                subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listfile), "-c", "copy", str(merged)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                convert_to_wav(str(merged), str(wav))

                # STT transcription
                transcript = transcribe_audio(str(wav))
                await ws.send_json({"type": "transcript", "text": transcript})

                # LLaMA evaluation
                prompt = f"""
Candidate answered: "{transcript}"

Provide:
1. Score (0–10)
2. Short Evaluation
3. Next Interview Question
"""
                llm_reply = ask_llama(prompt)

                # TTS
                tts_file = temp_dir / "tts.wav"
                make_tts(llm_reply, str(tts_file))

                try:
                    tts_b64 = base64.b64encode(tts_file.read_bytes()).decode()
                except:
                    tts_b64 = ""

                await ws.send_json({
                    "type": "llm",
                    "text": llm_reply,
                    "tts": tts_b64
                })

                session_log["exchanges"].append({
                    "transcript": transcript,
                    "response": llm_reply,
                    "timestamp": str(datetime.now())
                })

                chunks = []

            # ------------- END SESSION ------------ #
            elif mtype == "end_session":
                save_path = SESSIONS_DIR / f"{session_id}.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(session_log, f, indent=4)

                await ws.send_json({
                    "type": "bye",
                    "message": "Session saved",
                    "file": str(save_path)
                })
                break

    except WebSocketDisconnect:
        print("Client disconnected.")

    finally:
        # cleanup
        try:
            for f in temp_dir.glob("*"):
                f.unlink()
            temp_dir.rmdir()
        except:
            pass

