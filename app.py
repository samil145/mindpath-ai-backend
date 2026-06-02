from __future__ import annotations

import os
import re
import uuid
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None

load_dotenv()

app = FastAPI(title="MindPath AI Backend")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"
FIRESTORE_RETENTION_DAYS = int(os.getenv("FIRESTORE_RETENTION_DAYS", "7"))
DEFAULT_OWNER_ID = os.getenv("FIREBASE_DEFAULT_OWNER_ID", "demo-user")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
)

SYSTEM_PROMPT = """
You are MindPath, a safe mental wellbeing companion.

Rules:
- You are not a therapist, doctor, psychologist, or emergency service.
- Do not diagnose the user.
- Do not prescribe medicine.
- Do not give dangerous instructions.
- Keep replies short, calm, and practical.
- Give psychoeducation and simple coping support only.

When useful, recommend one of these exact tags:
[GAME:adhd_attention] for focus, distraction, ADHD, attention problems.
[GAME:anxiety_grounding] for anxiety, worry, stress, overthinking.
[GAME:panic_breathing] for panic attacks, breathing difficulty, intense fear.

If the backend marks the situation as crisis, do not generate a normal answer.
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def expiry_timestamp() -> datetime:
    return utc_now() + timedelta(days=FIRESTORE_RETENTION_DAYS)


def parse_service_account_json(raw_json: str) -> dict[str, Any]:
    data = json.loads(raw_json)
    private_key = data.get("private_key")
    if isinstance(private_key, str):
        data["private_key"] = private_key.replace("\\n", "\n")
    return data


def initialize_firestore():
    if firebase_admin is None or firestore is None or credentials is None:
        print("Firestore disabled: firebase-admin is not installed.")
        return None

    try:
        if not firebase_admin._apps:
            service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            project_id = os.getenv("FIREBASE_PROJECT_ID")
            use_application_default = os.getenv("FIREBASE_USE_APPLICATION_DEFAULT", "").lower() == "true"

            if service_account_json:
                cred = credentials.Certificate(parse_service_account_json(service_account_json))
                firebase_admin.initialize_app(cred)
            elif service_account_path:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
            elif use_application_default:
                options = {"projectId": project_id} if project_id else None
                firebase_admin.initialize_app(options=options)
            else:
                print("Firestore disabled: no Firebase credentials configured.")
                return None

        return firestore.client()
    except Exception as error:
        print("Firestore disabled:", error)
        return None


db = initialize_firestore()


def firestore_is_enabled() -> bool:
    return db is not None


def session_ref(session_id: str):
    if db is None:
        return None
    return db.collection("sessions").document(session_id)


def owner_for(user_id: Optional[str]) -> str:
    return user_id or DEFAULT_OWNER_ID


def upsert_session(
    session_id: str,
    owner_id: str,
    distress_level: str,
    escalation_level: str,
    crisis: bool
) -> None:
    ref = session_ref(session_id)
    if ref is None:
        return

    try:
        now = utc_now()
        data = {
            "session_id": session_id,
            "owner_id": owner_id,
            "updated_at": now,
            "distress_level": distress_level,
            "escalation_level": escalation_level,
            "crisis": crisis,
            "expires_at": expiry_timestamp(),
        }
        snapshot = ref.get()
        if not snapshot.exists:
            data["created_at"] = now
        ref.set(data, merge=True)
    except Exception as error:
        print("Firestore session write failed:", error)


def save_message(
    session_id: str,
    owner_id: str,
    sender: str,
    text: str,
    modality: str,
    distress_level: str,
    escalation_level: str,
    recommended_game: Optional[str],
    crisis: bool
) -> None:
    ref = session_ref(session_id)
    if ref is None:
        return

    try:
        upsert_session(session_id, owner_id, distress_level, escalation_level, crisis)
        ref.collection("messages").add({
            "sender": sender,
            "text": text,
            "modality": modality,
            "timestamp": utc_now(),
            "distress_level": distress_level,
            "escalation_level": escalation_level,
            "recommended_game": recommended_game,
            "crisis": crisis,
            "expires_at": expiry_timestamp(),
        })
    except Exception as error:
        print("Firestore message write failed:", error)


def save_game_completion(req: GameCompletionRequest) -> bool:
    ref = session_ref(req.session_id)
    if ref is None:
        return False

    try:
        owner_id = owner_for(req.user_id)
        now = utc_now()
        upsert_session(req.session_id, owner_id, "game", "engagement", False)
        ref.collection("game_events").add({
            "owner_id": owner_id,
            "game_id": req.game_id,
            "mission_id": req.mission_id,
            "completed": req.completed,
            "score": req.score,
            "timestamp": now,
            "expires_at": expiry_timestamp(),
        })

        if req.mission_id:
            ref.collection("missions").document(req.mission_id).set({
                "owner_id": owner_id,
                "game_id": req.game_id,
                "completed": req.completed,
                "score": req.score,
                "updated_at": now,
                "expires_at": expiry_timestamp(),
            }, merge=True)

        return True
    except Exception as error:
        print("Firestore game completion write failed:", error)
        return False


def delete_collection(collection_ref, batch_size: int = 50) -> int:
    deleted = 0
    while True:
        docs = list(collection_ref.limit(batch_size).stream())
        if not docs:
            return deleted

        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        deleted += len(docs)


def delete_session_documents(session_id: str) -> dict[str, Any]:
    ref = session_ref(session_id)
    if ref is None:
        return {
            "deleted": False,
            "session_id": session_id,
            "firebase_enabled": False,
            "message": "Firestore is not configured."
        }

    counts = {
        "messages": delete_collection(ref.collection("messages")),
        "game_events": delete_collection(ref.collection("game_events")),
        "missions": delete_collection(ref.collection("missions")),
    }
    ref.delete()
    return {
        "deleted": True,
        "session_id": session_id,
        "firebase_enabled": True,
        "deleted_documents": counts
    }


class MessageRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    modality: str = "text"
    user_id: Optional[str] = None


class MessageResponse(BaseModel):
    session_id: str
    reply: str
    distress_level: str
    escalation_level: str
    recommended_game: Optional[str] = None
    crisis: bool = False
    firebase_enabled: bool = False


class GameCompletionRequest(BaseModel):
    session_id: str
    game_id: str
    completed: bool = True
    score: Optional[int] = None
    mission_id: Optional[str] = None
    user_id: Optional[str] = None


CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bi want to die\b",
    r"\bwant to die\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bhurt myself\b",
    r"\bself[- ]harm\b",
    r"\boverdose\b",
]

HIGH_CONCERN_PATTERNS = [
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\bcan't cope\b",
    r"\bcannot cope\b",
    r"\bpanic attack\b",
    r"\bpanic\b",
    r"\bcrisis\b",
    r"\bi feel empty\b",
]

LOW_DISTRESS_PATTERNS = [
    r"\banxiety\b",
    r"\banxious\b",
    r"\bstress\b",
    r"\bstressed\b",
    r"\bworried\b",
    r"\bsad\b",
    r"\btired\b",
    r"\boverthinking\b",
    r"\bcan'?t focus\b",
    r"\bcannot focus\b",
    r"\bcan'?t concentrate\b",
    r"\bcannot concentrate\b",
    r"\bfocus\b",
    r"\battention\b",
    r"\bconcentration\b",
    r"\badhd\b",
    r"\bdistracted\b",
    r"\bocd\b",
    r"\bintrusive thoughts?\b",
    r"\bobsession(s)?\b",
    r"\bcompulsion(s)?\b",
    r"\bdepressed\b",
    r"\bdepression\b",
    r"\blonely\b",
    r"\bloneliness\b",
    r"\bsleep\b",
    r"\binsomnia\b",
]


def has_match(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def recommend_game(text: str) -> Optional[str]:
    if re.search(r"\bpanic\b|\bbreath\b|\bbreathing\b|\bcannot breathe\b", text):
        return "panic_breathing"

    if re.search(r"\banxiety\b|\banxious\b|\bworried\b|\bstress\b|\boverthinking\b", text):
        return "anxiety_grounding"

    if re.search(r"\badhd\b|\bfocus\b|\bdistracted\b|\battention\b", text):
        return "adhd_attention"

    return None


def normalize_text(message: str) -> str:
    return message.lower().replace("’", "'").strip()


def classify_message(message: str):
    text = normalize_text(message)

    if has_match(text, CRISIS_PATTERNS):
        return "T3", "crisis", None, True

    if has_match(text, HIGH_CONCERN_PATTERNS):
        return "T2", "high_concern", recommend_game(text), False

    if has_match(text, LOW_DISTRESS_PATTERNS):
        return "T1", "low_distress", recommend_game(text), False

    return "T1", "standard", None, False


def crisis_reply() -> str:
    return (
        "I’m really sorry you are feeling this way. "
        "This may be a crisis situation, so please contact local emergency services now "
        "or reach out to a trusted person near you immediately. "
        "MindPath is not a replacement for professional help."
    )


def fallback_reply(user_message: str, game: Optional[str]) -> str:
    text = normalize_text(user_message)

    if "ocd" in text or "intrusive" in text or "obsession" in text or "compulsion" in text:
        return (
            "OCD-like intrusive thoughts can feel repetitive and hard to ignore, but having a thought does not mean you want it or will act on it. "
            "Try labeling it as an intrusive thought, pause before doing any checking or reassurance behavior, and bring attention back to one concrete task. "
            "MindPath cannot diagnose OCD, so if this is affecting daily life, a qualified mental health professional can help."
        )

    if "depress" in text or "lonely" in text or "sad" in text:
        return (
            "I hear that this feels heavy. A small first step is to reduce the size of the task: drink water, sit somewhere safe, and message one trusted person if you can. "
            "If this low mood keeps coming back or affects daily life, professional support would be appropriate."
        )

    if "sleep" in text or "insomnia" in text or "tired" in text:
        return (
            "Sleep problems can make emotions feel stronger. For tonight, try a simple wind-down: dim the screen, slow your breathing, and write one worry on paper instead of solving it now. "
            "If sleep problems continue, consider speaking with a health professional."
        )

    if game == "panic_breathing":
        return "It sounds intense right now. A short breathing exercise may help. [GAME:panic_breathing]"

    if game == "anxiety_grounding":
        return "It sounds like your mind is overloaded. A grounding exercise may help. [GAME:anxiety_grounding]"

    if game == "adhd_attention":
        return "It sounds like focusing is difficult right now. A short attention exercise may help. [GAME:adhd_attention]"

    return (
        "I hear you. I can support with coping guidance, reflection, and simple next steps. "
        "Tell me what feels hardest right now, and I will suggest one practical step."
    )


def call_gemini(user_message: str, distress_level: str, escalation_level: str, game: Optional[str]) -> str:
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": SYSTEM_PROMPT
                    + f"\nBackend classification: {distress_level}, {escalation_level}."
                    + f"\nRecommended game: {game or 'none'}."
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_message}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 180
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            }
        ]
    }

    try:
        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return fallback_reply(user_message, game)

        if candidates[0].get("finishReason") == "SAFETY":
            return fallback_reply(user_message, game)

        parts = candidates[0]["content"]["parts"]
        answer = parts[0].get("text", "").strip()

        if not answer:
            return fallback_reply(user_message, game)

        return answer

    except Exception as error:
        print("Gemini error:", error)
        return fallback_reply(user_message, game)


@app.get("/")
def root():
    return {
        "status": "MindPath AI Backend running",
        "model": MODEL,
        "firebase_enabled": firestore_is_enabled()
    }


@app.post("/message", response_model=MessageResponse)
def message(req: MessageRequest):
    session_id = req.session_id or str(uuid.uuid4())
    owner_id = owner_for(req.user_id)

    distress_level, escalation_level, game, crisis = classify_message(req.message)

    if crisis:
        reply = crisis_reply()
    else:
        reply = call_gemini(req.message, distress_level, escalation_level, game)

    if game and f"[GAME:{game}]" not in reply:
        reply += f" [GAME:{game}]"

    save_message(
        session_id=session_id,
        owner_id=owner_id,
        sender="user",
        text=req.message,
        modality=req.modality,
        distress_level=distress_level,
        escalation_level=escalation_level,
        recommended_game=game,
        crisis=crisis
    )
    save_message(
        session_id=session_id,
        owner_id=owner_id,
        sender="assistant",
        text=reply,
        modality="text",
        distress_level=distress_level,
        escalation_level=escalation_level,
        recommended_game=game,
        crisis=crisis
    )

    return MessageResponse(
        session_id=session_id,
        reply=reply,
        distress_level=distress_level,
        escalation_level=escalation_level,
        recommended_game=game,
        crisis=crisis,
        firebase_enabled=firestore_is_enabled()
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    return delete_session_documents(session_id)


@app.post("/game-completion")
def game_completion(req: GameCompletionRequest):
    stored = save_game_completion(req)
    return {
        "stored": stored,
        "session_id": req.session_id,
        "game_id": req.game_id,
        "completed": req.completed,
        "firebase_enabled": firestore_is_enabled()
    }
