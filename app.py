import os
import re
import uuid
import requests
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MindPath AI Backend")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash-lite"

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


class MessageRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    modality: str = "text"


class MessageResponse(BaseModel):
    session_id: str
    reply: str
    distress_level: str
    escalation_level: str
    recommended_game: Optional[str] = None
    crisis: bool = False


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
    r"\bcan't focus\b",
    r"\bcannot focus\b",
    r"\badhd\b",
    r"\bdistracted\b",
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


def classify_message(message: str):
    text = message.lower().strip()

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


def fallback_reply(game: Optional[str]) -> str:
    if game == "panic_breathing":
        return "It sounds intense right now. A short breathing exercise may help. [GAME:panic_breathing]"

    if game == "anxiety_grounding":
        return "It sounds like your mind is overloaded. A grounding exercise may help. [GAME:anxiety_grounding]"

    if game == "adhd_attention":
        return "It sounds like focusing is difficult right now. A short attention exercise may help. [GAME:adhd_attention]"

    return "I hear you. Try describing what you are feeling in one or two sentences, and we can choose a small next step."


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
            return fallback_reply(game)

        if candidates[0].get("finishReason") == "SAFETY":
            return fallback_reply(game)

        parts = candidates[0]["content"]["parts"]
        answer = parts[0].get("text", "").strip()

        if not answer:
            return fallback_reply(game)

        return answer

    except Exception as error:
        print("Gemini error:", error)
        return fallback_reply(game)


@app.get("/")
def root():
    return {
        "status": "MindPath AI Backend running",
        "model": MODEL
    }


@app.post("/message", response_model=MessageResponse)
def message(req: MessageRequest):
    session_id = req.session_id or str(uuid.uuid4())

    distress_level, escalation_level, game, crisis = classify_message(req.message)

    if crisis:
        reply = crisis_reply()
    else:
        reply = call_gemini(req.message, distress_level, escalation_level, game)

    if game and f"[GAME:{game}]" not in reply:
        reply += f" [GAME:{game}]"

    return MessageResponse(
        session_id=session_id,
        reply=reply,
        distress_level=distress_level,
        escalation_level=escalation_level,
        recommended_game=game,
        crisis=crisis
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    return {
        "deleted": True,
        "session_id": session_id
    }
