import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import asyncio
import os
import subprocess
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="Intelligent Multimodal Edge AI Assistant",
    description="YOLO11 scene awareness with local Gemma inference.",
    version="1.2.0",
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_DIR / "vision" / "latest_scene.json"
SCENE_MAX_AGE_SECONDS = 5


class ChatRequest(BaseModel):
    message: str


def read_scene():
    """Read the latest YOLO11 snapshot and determine whether it is fresh."""
    try:
        with SCENE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        timestamp = data.get("timestamp")
        age_seconds = None

        if timestamp:
            try:
                scene_time = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
                if scene_time.tzinfo is None:
                    scene_time = scene_time.replace(tzinfo=timezone.utc)

                age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - scene_time).total_seconds(),
                )
            except (ValueError, TypeError):
                age_seconds = None

        original_status = data.get("status", "unknown")

        if original_status != "online":
            status = original_status
        elif age_seconds is None or age_seconds > SCENE_MAX_AGE_SECONDS:
            status = "stale"
        else:
            status = "online"

        objects = data.get("objects", [])
        if not isinstance(objects, list):
            objects = []

        return {
            "status": status,
            "source": data.get("source", "YOLO11"),
            "timestamp": timestamp,
            "fps": data.get("fps"),
            "objects": objects,
            "object_count": len(objects),
            "age_seconds": (
                round(age_seconds, 2) if age_seconds is not None else None
            ),
            "message": data.get("message"),
        }

    except (OSError, ValueError, TypeError) as exc:
        return {
            "status": "error",
            "source": "YOLO11",
            "timestamp": None,
            "fps": None,
            "objects": [],
            "object_count": 0,
            "age_seconds": None,
            "message": str(exc),
        }


def build_scene_context(scene):
    """Build honest camera context for the language model."""
    if scene["status"] != "online":
        return (
            "The YOLO11 camera snapshot is not currently confirmed live. "
            f"Scene status: {scene['status']}. "
            "Do not claim to know what is currently visible."
        )

    objects = scene["objects"]
    if not objects:
        object_description = (
            "YOLO11 detected no objects in this snapshot. "
            "This does not prove the scene is empty."
        )
    else:
        descriptions = []
        for obj in objects:
            label = obj.get("label", "unknown object")
            confidence = obj.get("confidence")
            if isinstance(confidence, (int, float)):
                descriptions.append(
                    f"{label} (confidence {confidence:.1%})"
                )
            else:
                descriptions.append(str(label))

        object_description = "; ".join(descriptions)

    return (
        f"Scene status: {scene['status']}\n"
        f"Snapshot timestamp: {scene['timestamp']}\n"
        f"Snapshot age in seconds: {scene['age_seconds']}\n"
        f"Detector FPS: {scene['fps']}\n"
        f"Detected object count: {scene['object_count']}\n"
        f"Detected objects: {object_description}\n"
        "The detections are the authoritative object labels for this "
        "snapshot. They do not establish exact distance, object identity, "
        "intent, or actions that the detector does not report."
    )


def get_label_counts(scene):
    """Count detected labels directly from the snapshot."""
    return Counter(
        str(obj.get("label", "unknown")).strip().lower()
        for obj in scene["objects"]
        if isinstance(obj, dict)
    )


def deterministic_object_answer(message, scene):
    """
    Answer clear object-count questions from Python, avoiding LLM errors.
    Return None when the question should be answered by Gemma.
    """
    question = message.lower().strip().rstrip("?!.")

    if scene["status"] != "online":
        camera_terms = (
            "camera", "see", "visible", "detected", "in front",
            "around me", "in the scene", "how many", "object",
            "person", "people", "what is here",
        )
        if any(term in question for term in camera_terms):
            return (
                "I can't reliably answer that from the camera right now "
                f"because the YOLO11 snapshot status is '{scene['status']}'. "
                "Please check that the detector is running and updating "
                "the scene snapshot."
            )

    # Handle explicit total-object-count questions.
    total_phrases = (
        "how many objects",
        "number of objects",
        "total objects",
        "object count",
        "how many things",
    )

    if any(phrase in question for phrase in total_phrases):
        return (
            f"The latest YOLO11 snapshot contains "
            f"{scene['object_count']} detected object(s)."
        )

    # Handle common class-count questions.
    label_aliases = {
        "people": ("person",),
        "persons": ("person",),
        "person": ("person",),
        "pedestrians": ("person",),
        "cars": ("car",),
        "automobiles": ("car",),
        "bikes": ("bicycle",),
        "bicycles": ("bicycle",),
        "motorbikes": ("motorcycle",),
        "motorcycles": ("motorcycle",),
        "dogs": ("dog",),
        "cats": ("cat",),
        "chairs": ("chair",),
        "bottles": ("bottle",),
        "laptops": ("laptop",),
        "phones": ("cell phone",),
        "cellphones": ("cell phone",),
        "cell phones": ("cell phone",),
        "suitcases": ("suitcase",),
        "buses": ("bus",),
        "trucks": ("truck",),
        "birds": ("bird",),
        "books": ("book",),
    }

    count_intents = (
        "how many",
        "count the",
        "count all",
        "number of",
        "how much",
    )

    asks_for_count = any(intent in question for intent in count_intents)
    if asks_for_count:
        for alias, labels in sorted(
            label_aliases.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if alias in question:
                counts = get_label_counts(scene)
                count = sum(counts.get(label, 0) for label in labels)
                label_name = labels[0]
                display_name = alias

                return (
                    f"The latest YOLO11 snapshot detected {count} "
                    f"{display_name}."
                )

        # Exact label counts for other labels already detected.
        counts = get_label_counts(scene)
        for label, count in sorted(
            counts.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if label and label in question:
                return (
                    f"The latest YOLO11 snapshot detected {count} "
                    f"{label}{'' if count == 1 else 's'}."
                )

    return None


@app.get("/ui", response_class=HTMLResponse)
def chat_ui():
    html_path = PROJECT_DIR / "frontend" / "index.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="Chat UI file not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/ui", response_class=HTMLResponse)
def chat_ui():
    html_path = PROJECT_DIR / "frontend" / "index.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="Chat UI file not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))



@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    process = None

    try:
        from vosk import Model, KaldiRecognizer

        model_path = PROJECT_DIR / "vosk-model-small-en-us-0.15"
        if not model_path.is_dir():
            await websocket.send_json({
                "type": "error",
                "text": f"Vosk model not found: {model_path}"
            })
            return

        model = Model(str(model_path))
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)

        source = "bluez_input.83_1F_4C_E8_EE_2D.0"
        process = subprocess.Popen(
            [
                "parec", "--raw",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                f"--device={source}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        await websocket.send_json({
            "type": "status",
            "text": "Listening through Airdopes microphone."
        })

        while True:
            chunk = await asyncio.to_thread(process.stdout.read, 4000)
            if not chunk:
                error = ""
                if process.poll() is not None and process.stderr:
                    error = process.stderr.read().decode(errors="replace")[-1000:]
                await websocket.send_json({
                    "type": "error",
                    "text": "Microphone capture stopped. " + error
                })
                break

            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    await websocket.send_json({"type": "final", "text": text})
            else:
                result = json.loads(recognizer.PartialResult())
                text = result.get("partial", "").strip()
                if text:
                    await websocket.send_json({"type": "partial", "text": text})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "text": str(exc)})
        except Exception:
            pass
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, 2)
                except Exception:
                    process.kill()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/")
def root():
    return {
        "service": "Intelligent Multimodal Edge AI Assistant",
        "version": app.version,
        "status": "running",
        "llm": MODEL,
        "endpoints": ["/scene", "/scene/summary", "/chat", "/docs"],
    }


@app.get("/scene")
def get_scene():
    """Return the latest detector snapshot and its freshness status."""
    return read_scene()


@app.get("/scene/summary")
def scene_summary():
    """Return a deterministic summary of the latest detector snapshot."""
    scene = read_scene()
    counts = get_label_counts(scene)

    return {
        "status": scene["status"],
        "timestamp": scene["timestamp"],
        "age_seconds": scene["age_seconds"],
        "object_count": scene["object_count"],
        "label_counts": dict(counts),
        "objects": scene["objects"],
        "message": scene["message"],
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """Answer general questions and camera-aware questions using local Gemma."""
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please provide a question or message.",
        )

    # Read one snapshot so answers and returned metadata refer to the same data.
    scene = read_scene()

    # Use exact Python counts for clearly identifiable counting questions.
    exact_answer = deterministic_object_answer(message, scene)
    if exact_answer is not None:
        return {
            "response": exact_answer,
            "answer_source": "deterministic_scene_logic",
            "scene_status": scene["status"],
            "scene_object_count": scene["object_count"],
            "scene_timestamp": scene["timestamp"],
            "scene_age_seconds": scene["age_seconds"],
        }

    scene_context = build_scene_context(scene)

    prompt = f"""You are a helpful, intelligent, general-purpose AI assistant
running locally on an NVIDIA Jetson Orin Nano.

You can answer general knowledge questions, explain concepts, help with
programming, solve problems, and answer questions about the camera scene.

CAMERA SNAPSHOT CONTEXT:
{scene_context}

IMPORTANT RULES:
1. Answer the user's actual question directly, clearly, and naturally.
2. For general questions unrelated to the camera, use your knowledge.
3. For camera-related questions, use the supplied YOLO11 snapshot.
4. Never invent objects or visual details that are not supported by the
   supplied snapshot.
5. If the snapshot is stale, unavailable, or in an error state, do not
   claim to know what is currently visible.
6. YOLO11 object detections alone do not establish exact distance, identity,
   intent, colors, text, or actions unless that information is supplied.
7. If the snapshot does not provide enough information, explain the
   limitation honestly.
8. Do not force camera context into unrelated general questions.
9. You do not have live internet access through this prompt. Do not claim
   to have checked current online information.
10. Match the user's requested language and level of detail when possible.

USER QUESTION:
{message}

ANSWER:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The local language model returned an empty answer.",
            )

        return {
            "response": answer,
            "answer_source": "gemma3:4b",
            "scene_status": scene["status"],
            "scene_object_count": scene["object_count"],
            "scene_timestamp": scene["timestamp"],
            "scene_age_seconds": scene["age_seconds"],
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not get a response from local Ollama. "
                "Check that Ollama is running and the model is available. "
                f"Details: {exc}"
            ),
        ) from exc
