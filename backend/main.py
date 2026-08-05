from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Query as FastAPIQuery,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
    import openai
except ImportError:  # Allows the non-AI features to run without the SDK.
    AsyncOpenAI = None  # type: ignore[assignment]
    openai = None  # type: ignore[assignment]

try:
    from .database import (
        activate_conversation,
        add_message,
        complete_reminder,
        create_profile,
        create_reminder,
        delete_conversation,
        delete_reminder,
        get_active_conversation_messages,
        get_conversation,
        get_conversation_messages,
        get_or_create_active_conversation,
        get_profile,
        get_recent_messages,
        get_reminder_state,
        hide_reminder,
        init_db,
        list_conversations,
        mark_reminder_notified,
        start_new_conversation,
        update_profile,
        update_reminder,
    )
except ImportError:
    from database import (  # type: ignore[no-redef]
        activate_conversation,
        add_message,
        complete_reminder,
        create_profile,
        create_reminder,
        delete_conversation,
        delete_reminder,
        get_active_conversation_messages,
        get_conversation,
        get_conversation_messages,
        get_or_create_active_conversation,
        get_profile,
        get_recent_messages,
        get_reminder_state,
        hide_reminder,
        init_db,
        list_conversations,
        mark_reminder_notified,
        start_new_conversation,
        update_profile,
        update_reminder,
    )


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
MAX_MESSAGE_LENGTH = 2000

client = None

if AsyncOpenAI and OPENAI_API_KEY and OPENAI_MODEL:
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=2,
        timeout=40.0,
    )


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Kelsie Backend",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "static",
    )
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

init_db()


# ============================================================
# REQUEST MODELS
# ============================================================


class ProfilePayload(BaseModel):
    user_id: Optional[str] = None
    id: Optional[str] = None
    name: str = ""
    mode: str = "both"
    timezone: str = "America/Toronto"
    daily_overview_enabled: bool = True
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    proactivity: str = "balanced"

    class Config:
        extra = "allow"


class ChatPayload(BaseModel):
    message: str
    user_id: Optional[str] = None


class ReminderCreatePayload(BaseModel):
    title: str
    scheduled_for: str
    alert_offset_minutes: int = 0


class ReminderUpdatePayload(BaseModel):
    title: Optional[str] = None
    scheduled_for: Optional[str] = None
    alert_offset_minutes: Optional[int] = None


class ReminderHidePayload(BaseModel):
    minutes: int = 15


# ============================================================
# GENERAL HELPERS
# ============================================================


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    model_dump = getattr(model, "model_dump", None)

    if callable(model_dump):
        return dict(model_dump())

    return dict(model.dict())


def clean_message(message: str) -> str:
    cleaned = str(message).strip()

    if len(cleaned) > MAX_MESSAGE_LENGTH:
        cleaned = cleaned[:MAX_MESSAGE_LENGTH]

    return cleaned


def contains_word(
    text: str,
    words: List[str],
) -> bool:
    pattern = r"\b(" + "|".join(
        re.escape(word)
        for word in words
    ) + r")\b"

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


def resolve_profile_user_id(
    profile_data: Dict[str, Any],
) -> str:
    user_id = (
        profile_data.get("user_id")
        or profile_data.get("id")
    )

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="A user_id is required.",
        )

    return str(user_id)


def build_system_prompt(
    profile: Optional[Dict[str, Any]],
) -> str:
    profile = profile or {}

    name = str(profile.get("name") or "the user")
    mode = str(profile.get("mode") or "both")
    timezone = str(
        profile.get("timezone")
        or "America/Toronto"
    )

    return f"""
You are Kelsie, a personal AI companion for students and
professionals. You should feel calm, observant, natural, and
helpful rather than robotic or overly enthusiastic.

Current user information:
- Name: {name}
- Primary mode: {mode}
- Timezone: {timezone}

Guidelines:
- Respond naturally and clearly.
- Keep responses suitable for a compact chat window.
- Remember the current conversation context.
- Do not repeatedly introduce yourself.
- Do not claim that you completed actions you cannot perform.
- Ask only necessary follow-up questions.
- Use plain text unless structure genuinely helps.
""".strip()


# ============================================================
# AI RESPONSE
# ============================================================


async def get_kelsie_reply(
    user_id: str,
    conversation_id: int,
    user_message: str,
) -> str:
    cleaned_message = clean_message(
        user_message
    )

    if not cleaned_message:
        return "Please enter a message."

    lowered_message = cleaned_message.lower()

    if contains_word(
        lowered_message,
        ["time"],
    ):
        return (
            "It’s "
            f"{datetime.now().strftime('%I:%M %p')} "
            "right now."
        )

    if contains_word(
        lowered_message,
        ["date", "day"],
    ):
        return (
            "Today is "
            f"{datetime.now().strftime('%B %d, %Y')}."
        )

    if client is None:
        return (
            "My AI connection isn’t configured yet, "
            "but your message has been saved."
        )

    profile = get_profile(user_id)

    recent_messages = get_recent_messages(
        conversation_id,
        limit=20,
    )

    model_messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(profile),
        }
    ]

    for message in recent_messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()

        if role not in {"user", "assistant"} or not content:
            continue

        model_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=model_messages,
            temperature=0.7,
            max_tokens=350,
        )

        reply = response.choices[0].message.content

        if not reply:
            return (
                "I couldn’t generate a response just now, "
                "but your message has been saved."
            )

        return str(reply).strip()

    except Exception as error:
        print(
            f"AI provider error: {error}"
        )

        return (
            "My AI connection is unavailable right now, "
            "but your message has been saved."
        )


# ============================================================
# PROFILE ENDPOINTS
# ============================================================


def normalize_profile_payload(
    payload: ProfilePayload,
    path_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    profile_data = model_to_dict(payload)

    nested_profile = profile_data.get("profile")

    if isinstance(nested_profile, dict):
        profile_data.update(nested_profile)

    if path_user_id:
        profile_data["user_id"] = str(path_user_id)
        profile_data["id"] = str(path_user_id)

    if not profile_data.get("name"):
        display_name = profile_data.get("display_name")

        if display_name:
            profile_data["name"] = str(display_name)

    return profile_data


async def save_profile(
    payload: ProfilePayload,
    path_user_id: Optional[str] = None,
):
    profile_data = normalize_profile_payload(
        payload,
        path_user_id,
    )

    user_id = resolve_profile_user_id(profile_data)

    profile_data["user_id"] = user_id
    profile_data["id"] = user_id

    return create_profile(profile_data)


@app.post("/api/profiles")
@app.post(
    "/api/profile",
    include_in_schema=False,
)
@app.post(
    "/profiles",
    include_in_schema=False,
)
@app.post(
    "/profile",
    include_in_schema=False,
)
async def create_or_replace_profile(
    payload: ProfilePayload,
):
    return await save_profile(payload)


@app.post(
    "/api/profiles/{user_id}",
    include_in_schema=False,
)
@app.post(
    "/api/profile/{user_id}",
    include_in_schema=False,
)
@app.post(
    "/profiles/{user_id}",
    include_in_schema=False,
)
@app.post(
    "/profile/{user_id}",
    include_in_schema=False,
)
async def create_or_replace_profile_for_user(
    user_id: str,
    payload: ProfilePayload,
):
    return await save_profile(
        payload,
        user_id,
    )


async def find_profile(
    user_id: str,
):
    # Returning JSON null for a new user keeps the onboarding
    # flow from treating a missing profile as a server failure.
    return get_profile(user_id)


@app.get("/api/profiles/{user_id}")
@app.get(
    "/api/profile/{user_id}",
    include_in_schema=False,
)
@app.get(
    "/profiles/{user_id}",
    include_in_schema=False,
)
@app.get(
    "/profile/{user_id}",
    include_in_schema=False,
)
async def read_profile(user_id: str):
    return await find_profile(user_id)


@app.get(
    "/api/profiles",
    include_in_schema=False,
)
@app.get(
    "/api/profile",
    include_in_schema=False,
)
@app.get(
    "/profiles",
    include_in_schema=False,
)
@app.get(
    "/profile",
    include_in_schema=False,
)
async def read_profile_from_query(
    user_id: str = FastAPIQuery(...),
):
    return await find_profile(user_id)


async def edit_profile_from_payload(
    payload: ProfilePayload,
):
    profile_data = normalize_profile_payload(payload)
    user_id = resolve_profile_user_id(profile_data)

    profile_data.pop("user_id", None)
    profile_data.pop("id", None)

    return update_profile(
        user_id,
        profile_data,
    )


@app.put(
    "/api/profiles",
    include_in_schema=False,
)
@app.patch(
    "/api/profiles",
    include_in_schema=False,
)
@app.put(
    "/api/profile",
    include_in_schema=False,
)
@app.patch(
    "/api/profile",
    include_in_schema=False,
)
@app.put(
    "/profiles",
    include_in_schema=False,
)
@app.patch(
    "/profiles",
    include_in_schema=False,
)
@app.put(
    "/profile",
    include_in_schema=False,
)
@app.patch(
    "/profile",
    include_in_schema=False,
)
async def edit_profile_without_path(
    payload: ProfilePayload,
):
    return await edit_profile_from_payload(payload)


async def apply_profile_update(
    user_id: str,
    payload: ProfilePayload,
):
    profile_data = normalize_profile_payload(
        payload,
        user_id,
    )

    profile_data.pop("user_id", None)
    profile_data.pop("id", None)

    return update_profile(
        user_id,
        profile_data,
    )


@app.put("/api/profiles/{user_id}")
@app.patch("/api/profiles/{user_id}")
@app.put(
    "/api/profile/{user_id}",
    include_in_schema=False,
)
@app.patch(
    "/api/profile/{user_id}",
    include_in_schema=False,
)
@app.put(
    "/profiles/{user_id}",
    include_in_schema=False,
)
@app.patch(
    "/profiles/{user_id}",
    include_in_schema=False,
)
@app.put(
    "/profile/{user_id}",
    include_in_schema=False,
)
@app.patch(
    "/profile/{user_id}",
    include_in_schema=False,
)
async def edit_profile(
    user_id: str,
    payload: ProfilePayload,
):
    return await apply_profile_update(
        user_id,
        payload,
    )


# ============================================================
# CONVERSATION ENDPOINTS
# ============================================================


@app.get("/api/conversations/{user_id}")
async def read_conversations(
    user_id: str,
    limit: int = FastAPIQuery(
        default=50,
        ge=1,
        le=100,
    ),
):
    conversations = list_conversations(
        user_id,
        limit=limit,
    )

    active_conversation_id = next(
        (
            conversation["id"]
            for conversation in conversations
            if conversation["is_active"]
        ),
        None,
    )

    return {
        "active_conversation_id": active_conversation_id,
        "conversations": conversations,
    }


@app.get("/api/conversations/{user_id}/messages")
async def read_active_conversation_messages(
    user_id: str,
    limit: int = FastAPIQuery(
        default=100,
        ge=1,
        le=200,
    ),
):
    return get_active_conversation_messages(
        user_id,
        limit=limit,
    )


@app.post("/api/conversations/{user_id}/new")
async def create_new_conversation(user_id: str):
    conversation_id = start_new_conversation(
        user_id
    )

    return {
        "conversation_id": conversation_id,
        "conversation": get_conversation(
            user_id,
            conversation_id,
        ),
        "messages": [],
    }


@app.get(
    "/api/conversations/{user_id}/{conversation_id}/messages"
)
async def read_selected_conversation_messages(
    user_id: str,
    conversation_id: int,
    limit: int = FastAPIQuery(
        default=100,
        ge=1,
        le=200,
    ),
):
    try:
        messages = get_conversation_messages(
            user_id,
            conversation_id,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "conversation_id": conversation_id,
        "conversation": get_conversation(
            user_id,
            conversation_id,
        ),
        "messages": messages,
    }


@app.post(
    "/api/conversations/{user_id}/{conversation_id}/activate"
)
async def select_conversation(
    user_id: str,
    conversation_id: int,
):
    try:
        conversation = activate_conversation(
            user_id,
            conversation_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "conversation_id": conversation_id,
        "conversation": conversation,
        "messages": get_conversation_messages(
            user_id,
            conversation_id,
            limit=200,
        ),
    }


@app.delete(
    "/api/conversations/{user_id}/{conversation_id}"
)
async def remove_conversation(
    user_id: str,
    conversation_id: int,
):
    try:
        result = delete_conversation(
            user_id,
            conversation_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    active_conversation_id = int(
        result["active_conversation_id"]
    )

    return {
        **result,
        "messages": get_conversation_messages(
            user_id,
            active_conversation_id,
            limit=200,
        ),
    }


# ============================================================
# REMINDER ENDPOINTS
# ============================================================


@app.get("/api/reminders/{user_id}")
async def read_reminders(user_id: str):
    return get_reminder_state(user_id)


@app.post("/api/reminders/{user_id}")
async def add_reminder(
    user_id: str,
    payload: ReminderCreatePayload,
):
    try:
        reminder = create_reminder(
            user_id=user_id,
            title=payload.title,
            scheduled_for=payload.scheduled_for,
            alert_offset_minutes=payload.alert_offset_minutes,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "reminder": reminder,
        "state": get_reminder_state(user_id),
    }


@app.patch("/api/reminders/{user_id}/{reminder_id}")
async def edit_reminder(
    user_id: str,
    reminder_id: int,
    payload: ReminderUpdatePayload,
):
    if (
        payload.title is None
        and payload.scheduled_for is None
        and payload.alert_offset_minutes is None
    ):
        raise HTTPException(
            status_code=400,
            detail="No reminder changes were provided.",
        )

    try:
        reminder = update_reminder(
            user_id=user_id,
            reminder_id=reminder_id,
            title=payload.title,
            scheduled_for=payload.scheduled_for,
            alert_offset_minutes=payload.alert_offset_minutes,
        )
    except ValueError as error:
        status_code = (
            404
            if str(error) == "Reminder not found."
            else 400
        )
        raise HTTPException(
            status_code=status_code,
            detail=str(error),
        ) from error

    return {
        "reminder": reminder,
        "state": get_reminder_state(user_id),
    }


@app.post("/api/reminders/{user_id}/{reminder_id}/complete")
async def finish_reminder(
    user_id: str,
    reminder_id: int,
):
    try:
        reminder = complete_reminder(
            user_id,
            reminder_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "reminder": reminder,
        "state": get_reminder_state(user_id),
    }


@app.post("/api/reminders/{user_id}/{reminder_id}/hide")
async def temporarily_hide_reminder(
    user_id: str,
    reminder_id: int,
    payload: ReminderHidePayload,
):
    try:
        reminder = hide_reminder(
            user_id,
            reminder_id,
            minutes=payload.minutes,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "reminder": reminder,
        "state": get_reminder_state(user_id),
    }


@app.post("/api/reminders/{user_id}/{reminder_id}/notified")
async def record_reminder_notification(
    user_id: str,
    reminder_id: int,
):
    try:
        reminder = mark_reminder_notified(
            user_id,
            reminder_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "reminder": reminder,
    }


@app.delete("/api/reminders/{user_id}/{reminder_id}")
async def remove_reminder(
    user_id: str,
    reminder_id: int,
):
    try:
        result = delete_reminder(
            user_id,
            reminder_id,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        **result,
        "state": get_reminder_state(user_id),
    }


# ============================================================
# REST CHAT
# ============================================================


@app.post("/chat")
async def chat(payload: ChatPayload):
    user_id = str(
        payload.user_id
        or "anonymous-rest-user"
    )

    message = clean_message(
        payload.message
    )

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    conversation_id = get_or_create_active_conversation(
        user_id
    )

    add_message(
        conversation_id,
        "user",
        message,
    )

    reply = await get_kelsie_reply(
        user_id,
        conversation_id,
        message,
    )

    add_message(
        conversation_id,
        "assistant",
        reply,
    )

    return {
        "reply": reply,
        "conversation_id": conversation_id,
    }


# ============================================================
# WEBSOCKET
# ============================================================


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(
        self,
        websocket: WebSocket,
    ) -> None:
        await websocket.accept()

        self.active_connections.append(
            websocket
        )

    def disconnect(
        self,
        websocket: WebSocket,
    ) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(
                websocket
            )


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
):
    user_id = str(
        websocket.query_params.get("user_id")
        or "anonymous-websocket-user"
    )

    await manager.connect(websocket)

    try:
        get_or_create_active_conversation(
            user_id
        )

        while True:
            raw_message = await websocket.receive_text()

            user_message = clean_message(
                raw_message
            )

            if not user_message:
                await websocket.send_json(
                    {
                        "type": "assistant",
                        "message": "Please enter a message.",
                    }
                )

                continue

            # Resolve the active conversation for every message.
            # This makes history switching work without caching an
            # old conversation ID inside the WebSocket connection.
            conversation_id = (
                get_or_create_active_conversation(
                    user_id
                )
            )

            add_message(
                conversation_id,
                "user",
                user_message,
            )

            reply = await get_kelsie_reply(
                user_id,
                conversation_id,
                user_message,
            )

            add_message(
                conversation_id,
                "assistant",
                reply,
            )

            await websocket.send_json(
                {
                    "type": "assistant",
                    "message": reply,
                    "conversation_id": conversation_id,
                }
            )

    except WebSocketDisconnect:
        pass

    except Exception as error:
        print(
            f"WebSocket error: {error}"
        )

        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": (
                        "Kelsie encountered an error. "
                        "Please try again."
                    ),
                }
            )
        except Exception:
            pass

    finally:
        manager.disconnect(websocket)


# ============================================================
# HEALTH
# ============================================================


@app.get("/")
async def root():
    return {
        "status": "Kelsie backend is running",
        "widget": "/static/widget.html",
        "websocket": "/ws?user_id=YOUR_USER_ID",
        "ai_configured": client is not None,
        "model": OPENAI_MODEL or None,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ai_configured": client is not None,
    }


# Run from the project root:
# python3 -m uvicorn backend.main:app --reload