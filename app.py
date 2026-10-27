import base64
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
VISION_MODEL = "meta-llama/Llama-3.2-11b-Vision-Instruct"
REASONING_MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DATABASE_PATH = Path(
    os.getenv("GLOWBOT_DB_PATH", str(Path(__file__).with_name("glowbot_records.sqlite3")))
)
METRIC_LABELS = {
    "pigmentation_appearance": "Pigmentation appearance",
    "acne_like_appearance": "Acne-like appearance",
    "visible_scarring": "Visible scarring",
    "fine_line_appearance": "Fine-line appearance",
}


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key or api_key == "I will provide later":
        raise ValueError("Set a valid NEBIUS_API_KEY in the root .env file.")
    return OpenAI(api_key=api_key, base_url=NEBIUS_BASE_URL)


def get_database_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    connection = get_database_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                visual_output TEXT NOT NULL,
                guidance TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_profile_date "
            "ON analyses(profile_id, created_at)"
        )
        connection.commit()
    finally:
        connection.close()


def validate_profile_id(profile_id: str) -> str:
    normalized_id = profile_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]{2,32}", normalized_id):
        raise ValueError("Use 2-32 letters, numbers, hyphens, or underscores for the profile ID.")
    return normalized_id


def encode_image(image_file: Any) -> str:
    if image_file.size > MAX_IMAGE_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")

    try:
        with Image.open(image_file) as source_image:
            image = source_image.convert("RGB")
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
    except Exception as error:
        raise ValueError("The selected file could not be decoded as an image.") from error

    encoded_image = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


def parse_model_json(content: str, stage_name: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate_lines = candidate.splitlines()
        if candidate_lines and candidate_lines[0].startswith("```"):
            candidate_lines = candidate_lines[1:]
        if candidate_lines and candidate_lines[-1].strip() == "```":
            candidate_lines = candidate_lines[:-1]
        candidate = "\n".join(candidate_lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"{stage_name} did not return a JSON object.")
    try:
        parsed_output = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"{stage_name} returned invalid JSON.") from error
    if not isinstance(parsed_output, dict):
        raise ValueError(f"{stage_name} response must be a JSON object.")
    return parsed_output


def normalize_visual_output(raw_output: dict[str, Any]) -> dict[str, Any]:
    raw_metrics = raw_output.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    metrics: dict[str, dict[str, Any]] = {}
    for metric_id in METRIC_LABELS:
        raw_metric = raw_metrics.get(metric_id, {})
        if not isinstance(raw_metric, dict):
            raw_metric = {}
        score = raw_metric.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            score = None
        elif score < 0 or score > 4:
            score = None
        else:
            score = round(score, 1)
        confidence = raw_metric.get("confidence", "low")
        if not isinstance(confidence, str) or confidence not in {"low", "medium", "high"}:
            confidence = "low"
        metrics[metric_id] = {"score": score, "confidence": confidence}

    observations = raw_output.get("observations", [])
    limitations = raw_output.get("limitations", [])
    if not isinstance(observations, list):
        observations = []
    if not isinstance(limitations, list):
        limitations = []
    return {
        "observations": observations,
        "metrics": metrics,
        "limitations": limitations,
        "not_a_diagnosis": True,
    }


def process_visual_data(
    client: OpenAI, image_data_url: str, user_request: str
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0.1,
        max_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious visual wellness assistant, not a clinician. "
                    "Describe only visible appearance; do not identify a person, infer "
                    "age, ethnicity, health status, or diagnose acne, scars, or any "
                    "condition. Image lighting, camera processing, skin tone, and "
                    "image quality can bias estimates. Return only one JSON object with "
                    "keys: observations (array of objects with area, description, "
                    "confidence), metrics (object with pigmentation_appearance, "
                    "acne_like_appearance, visible_scarring, fine_line_appearance; "
                    "each has score and confidence), limitations (array of strings). "
                    "Each score is a rough visible-appearance estimate from 0 to 4, "
                    "where 0 means not apparent in this image and 4 means more "
                    "apparent. Use null when not assessable. Confidence must be low, "
                    "medium, or high. Never frame the score as a diagnosis."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_request},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The visual model returned an empty response.")
    return normalize_visual_output(parse_model_json(content, "Stage 1 visual analysis"))


def evaluate_visual_output(
    client: OpenAI, visual_output: dict[str, Any], user_request: str
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        temperature=0.2,
        max_tokens=1400,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful skincare and wellness information assistant, "
                    "not a doctor. Use only Stage 1's cautious visual observations as "
                    "context; do not diagnose or promise a result. Return only JSON "
                    "with keys summary (string), topical_care (array of strings), "
                    "food_habits (array of strings), general_wellness (array of strings), "
                    "face_yoga (array of objects "
                    "with name, steps, and caution), and clinician_guidance (string). "
                    "Prefer simple, low-risk, evidence-aware routines. Do not suggest "
                    "lemon juice, baking soda, toothpaste, undiluted essential oils, "
                    "or ingesting topical ingredients. Avoid supplement prescriptions, "
                    "extreme diets, or claims that a particular food treats skin or that "
                    "face yoga changes anatomy. Food suggestions must be ordinary "
                    "balanced-meal ideas, not treatment claims. Include patch testing "
                    "and sunscreen when relevant. "
                    "For persistent, painful, rapidly changing, or concerning findings, "
                    "recommend a licensed clinician."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User focus: {user_request}\n\n"
                    "Stage 1 structured visual output:\n"
                    f"{json.dumps(visual_output, ensure_ascii=True, indent=2)}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The reasoning model returned an empty response.")
    return parse_model_json(content, "Stage 2 reasoning")


def run_pipeline(image_data_url: str, user_request: str) -> dict[str, Any]:
    client = get_client()
    visual_output = process_visual_data(client, image_data_url, user_request)
    guidance = evaluate_visual_output(client, visual_output, user_request)
    return {"visual_output": visual_output, "guidance": guidance}


def answer_follow_up(
    visual_output: dict[str, Any],
    guidance: dict[str, Any],
    conversation: list[dict[str, str]],
    question: str,
) -> str:
    client = get_client()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a cautious skincare and wellness information chatbot. This is "
                "not medical advice. Do not diagnose, promise results, infer protected "
                "traits, or recommend risky DIY topical treatments, supplements, or "
                "extreme diets. For persistent, painful, rapidly changing, or "
                "concerning symptoms, recommend a licensed clinician. Answer from the "
                "provided scan context and be clear about uncertainty."
            ),
        },
        {
            "role": "user",
            "content": (
                "Context from the latest scan. Stage 1 visible observations:\n"
                f"{json.dumps(visual_output, ensure_ascii=True)}\n\n"
                "Stage 2 wellness guidance:\n"
                f"{json.dumps(guidance, ensure_ascii=True)}"
            ),
        },
    ]
    messages.extend(conversation)
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        temperature=0.3,
        max_tokens=900,
        messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The chatbot returned an empty response.")
    return content


def save_analysis(
    profile_id: str,
    visual_output: dict[str, Any],
    guidance: dict[str, Any],
) -> None:
    connection = get_database_connection()
    try:
        connection.execute(
            """
            INSERT INTO analyses (profile_id, created_at, visual_output, guidance)
            VALUES (?, ?, ?, ?)
            """,
            (
                profile_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(visual_output, ensure_ascii=True),
                json.dumps(guidance, ensure_ascii=True),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def get_analyses(profile_id: str) -> list[dict[str, Any]]:
    connection = get_database_connection()
    try:
        rows = connection.execute(
            """
            SELECT id, created_at, visual_output, guidance
            FROM analyses
            WHERE profile_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (profile_id,),
        ).fetchall()
    finally:
        connection.close()

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "visual_output": json.loads(row["visual_output"]),
                "guidance": json.loads(row["guidance"]),
            }
        )
    return records


def delete_profile_analyses(profile_id: str) -> None:
    connection = get_database_connection()
    try:
        connection.execute("DELETE FROM analyses WHERE profile_id = ?", (profile_id,))
        connection.commit()
    finally:
        connection.close()


def render_guidance(guidance: dict[str, Any]) -> None:
    st.markdown("#### Summary")
    st.write(guidance.get("summary", "No summary was provided."))

    care_sections = [
        ("Simple topical care", "topical_care"),
        ("Balanced food habits", "food_habits"),
        ("General wellness", "general_wellness"),
    ]
    for heading, key in care_sections:
        st.markdown(f"#### {heading}")
        recommendations = guidance.get(key, [])
        if isinstance(recommendations, list) and recommendations:
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        else:
            st.caption("No suggestions were returned for this section.")

    st.markdown("#### Gentle face yoga")
    yoga_steps = guidance.get("face_yoga", [])
    if isinstance(yoga_steps, list) and yoga_steps:
        for exercise in yoga_steps:
            if isinstance(exercise, dict):
                st.markdown(f"**{exercise.get('name', 'Suggested exercise')}**")
                st.write(exercise.get("steps", ""))
                caution = exercise.get("caution", "")
                if caution:
                    st.caption(caution)
            else:
                st.write(exercise)
    else:
        st.caption("No face yoga suggestions were returned.")

    st.markdown("#### When to seek professional advice")
    st.write(
        guidance.get(
            "clinician_guidance",
            "Consult a licensed clinician for persistent or concerning symptoms.",
        )
    )


def render_metric_chart(records: list[dict[str, Any]]) -> None:
    figure = go.Figure()
    for metric_id, label in METRIC_LABELS.items():
        dates: list[str] = []
        scores: list[float | None] = []
        for record in records:
            date_label = datetime.fromisoformat(record["created_at"]).astimezone().strftime(
                "%Y-%m-%d %H:%M"
            )
            metric = record["visual_output"].get("metrics", {}).get(metric_id, {})
            score = metric.get("score") if isinstance(metric, dict) else None
            dates.append(date_label)
            scores.append(score)
        figure.add_trace(
            go.Scatter(x=dates, y=scores, mode="lines+markers", name=label)
        )
    figure.update_layout(
        height=390,
        xaxis_title="Scan date",
        yaxis_title="Estimated visible appearance (0-4)",
        yaxis=dict(range=[0, 4], dtick=1),
        legend_title_text="Appearance measure",
        margin=dict(l=12, r=12, t=24, b=12),
    )
    st.plotly_chart(figure, use_container_width=True)


st.set_page_config(
    page_title="Varnika GlowBot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv()
initialize_database()

st.title("Varnika GlowBot")
st.caption("A private-by-default visual wellness journal powered by Nebius Token Factory.")

with st.sidebar:
    st.subheader("Your journal")
    entered_profile_id = st.text_input(
        "Profile ID", value="my-profile", max_chars=32,
        help="Use a pseudonym, not your name or contact details.",
    )
    try:
        profile_id = validate_profile_id(entered_profile_id)
        profile_error = ""
    except ValueError as error:
        profile_id = ""
        profile_error = str(error)
    if profile_error:
        st.warning(profile_error)
    st.divider()
    st.subheader("Model pipeline")
    st.markdown(f"**1 · Visual analysis**  \n`{VISION_MODEL}`")
    st.markdown(f"**2 · NVIDIA reasoning**  \n`{REASONING_MODEL}`")
    st.divider()
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if api_key and api_key != "I will provide later":
        st.success("Nebius API key detected")
    else:
        st.warning("Set NEBIUS_API_KEY in .env before running a scan")
    st.caption("Photos are sent to Nebius for inference; the local journal stores no photos.")

scan_tab, history_tab, about_tab = st.tabs(["New scan", "Progress history", "About"])

with scan_tab:
    st.subheader("Take or choose a photo")
    source_choice = st.radio(
        "Photo source", ["Use camera", "Upload photo"], horizontal=True
    )
    if source_choice == "Use camera":
        selected_image = st.camera_input("Capture a face photo")
    else:
        selected_image = st.file_uploader(
            "Choose an image", type=["png", "jpg", "jpeg", "webp"]
        )

    focus = st.text_area(
        "What would you like to focus on?",
        placeholder="Optional: pigmentation appearance, breakouts, fine lines, or a simple routine",
        max_chars=500,
    )
    adult_and_consent = st.checkbox(
        "I am 18 or older, and this is my photo or I have permission to submit it.",
        value=False,
    )
    data_consent = st.checkbox(
        "I understand this photo will be sent to Nebius for AI analysis and will not be stored in this app.",
        value=False,
    )

    if selected_image is not None:
        st.image(selected_image, caption="Photo selected for this scan", width=360)

    run_clicked = st.button(
        "Analyze photo", type="primary", disabled=selected_image is None or not profile_id
    )
    if run_clicked and selected_image is not None:
        if not adult_and_consent or not data_consent:
            st.error("Confirm both age/photo permission and data processing consent to continue.")
        else:
            focus_text = focus.strip() or "Give a general, cautious visual skincare overview."
            request_text = (
                f"Requested focus: {focus_text}. Use a consistent appearance scale for "
                "longitudinal journaling and state uncertainty clearly."
            )
            try:
                image_data_url = encode_image(selected_image)
                with st.spinner("Stage 1 is analyzing the image; Stage 2 is preparing guidance..."):
                    result = run_pipeline(image_data_url, request_text)
                    save_analysis(
                        profile_id,
                        result["visual_output"],
                        result["guidance"],
                    )
                st.session_state["pipeline_result"] = {
                    "profile_id": profile_id,
                    **result,
                }
                st.session_state["chat_history"] = []
                st.success("Analysis complete and saved to this profile's local journal.")
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(
                    f"Analysis could not be completed ({type(error).__name__}). "
                    "Check the Nebius key, model access, and network connection."
                )

    st.caption(
        "This is informational only, not a diagnosis or treatment. Photos are processed "
        "by the hosted model service. Use consistent lighting and camera distance for comparisons."
    )

    current_result = st.session_state.get("pipeline_result", {})
    if current_result and current_result.get("profile_id") == profile_id:
        st.divider()
        st.subheader("Visual observations")
        visual_output = current_result["visual_output"]
        for metric_id, label in METRIC_LABELS.items():
            metric = visual_output["metrics"].get(metric_id, {})
            score = metric.get("score")
            confidence = metric.get("confidence", "low")
            display_score = "Not assessable" if score is None else f"{score:g} / 4"
            st.write(f"**{label}:** {display_score} · {confidence} confidence")
        observations = visual_output.get("observations", [])
        if observations:
            st.markdown("#### Notes")
            st.json(observations)
        limitations = visual_output.get("limitations", [])
        if limitations:
            st.markdown("#### Image limitations")
            for limitation in limitations:
                st.markdown(f"- {limitation}")
        st.divider()
        st.subheader("Your wellness suggestions")
        render_guidance(current_result["guidance"])
        st.divider()
        st.subheader("Ask a follow-up")
        chat_history = st.session_state.setdefault("chat_history", [])
        for chat_message in chat_history:
            with st.chat_message(chat_message["role"]):
                st.markdown(chat_message["content"])
        with st.form("follow_up_form", clear_on_submit=True):
            follow_up_question = st.text_input(
                "Question about this scan or routine",
                max_chars=500,
                placeholder="Ask about a suggested step or how to make a routine gentler",
            )
            follow_up_submitted = st.form_submit_button("Send question")
        if follow_up_submitted and follow_up_question.strip():
            try:
                with st.spinner("Preparing a response..."):
                    follow_up_answer = answer_follow_up(
                        current_result["visual_output"],
                        current_result["guidance"],
                        chat_history,
                        follow_up_question.strip(),
                    )
                chat_history.extend(
                    [
                        {"role": "user", "content": follow_up_question.strip()},
                        {"role": "assistant", "content": follow_up_answer},
                    ]
                )
                st.rerun()
            except Exception as error:
                st.error(
                    f"The follow-up could not be answered ({type(error).__name__}). "
                    "Check Nebius model access and network connection."
                )

with history_tab:
    st.subheader("Progress over time")
    st.write(
        "These are model-estimated appearance scores, not clinical measurements. "
        "Lower scores mean the feature appeared less visible in that individual image; "
        "lighting, angle, expression, and camera processing can change estimates."
    )
    if profile_id:
        records = get_analyses(profile_id)
        if records:
            st.caption(f"{len(records)} saved scan(s) for profile `{profile_id}`")
            first_scan = datetime.fromisoformat(records[0]["created_at"])
            latest_scan = datetime.fromisoformat(records[-1]["created_at"])
            elapsed_days = max(0, (latest_scan - first_scan).days)
            st.markdown(f"**Tracking period:** {elapsed_days} day(s) across {len(records)} scan(s)")
            if len(records) > 1:
                st.markdown("#### Change since first scan")
                st.caption(
                    "Negative change means the feature appeared less visible in the latest "
                    "image; it is not proof of clinical improvement."
                )
                metric_columns = st.columns(len(METRIC_LABELS))
                for column, (metric_id, label) in zip(metric_columns, METRIC_LABELS.items()):
                    first_metric = records[0]["visual_output"].get("metrics", {}).get(metric_id, {})
                    latest_metric = records[-1]["visual_output"].get("metrics", {}).get(metric_id, {})
                    first_score = first_metric.get("score") if isinstance(first_metric, dict) else None
                    latest_score = latest_metric.get("score") if isinstance(latest_metric, dict) else None
                    if first_score is None or latest_score is None:
                        column.metric(label, "Not comparable")
                    else:
                        delta = round(latest_score - first_score, 1)
                        column.metric(label, f"{latest_score:g} / 4", f"{delta:+g} points")
            render_metric_chart(records)
            latest_record = records[-1]
            with st.expander("Latest scan details", expanded=False):
                st.write(
                    "Recorded: "
                    + datetime.fromisoformat(latest_record["created_at"])
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M %Z")
                )
                st.json(latest_record["visual_output"])
                render_guidance(latest_record["guidance"])
            st.divider()
            confirm_delete = st.checkbox("I understand this permanently deletes this profile's local history.")
            if st.button("Delete this profile's history", disabled=not confirm_delete):
                delete_profile_analyses(profile_id)
                st.session_state.pop("pipeline_result", None)
                st.success("Profile history deleted.")
                st.rerun()
        else:
            st.info("No scans saved for this profile yet. Complete a scan to start tracking.")
    else:
        st.info("Enter a valid profile ID in the sidebar to view its history.")

with about_tab:
    st.subheader("How the analysis works")
    st.markdown(
        f"1. **Visual observations:** `{VISION_MODEL}` receives the selected image and "
        "returns visible, non-diagnostic observations plus rough 0-4 appearance estimates.\n"
        f"2. **Guidance:** `{REASONING_MODEL}` evaluates only that structured output and "
        "returns cautious topical-care, general-wellness, and face-yoga suggestions.\n"
        "3. **Progress journal:** the app saves a timestamp, image fingerprint, model "
        "observations, and guidance in a local SQLite database. It does not save the image."
    )
    st.warning(
        "The score is an AI estimate, not a validated skin test. This prototype does not "
        "diagnose or treat acne, pigmentation disorders, scarring, or aging. Face yoga "
        "benefits are uncertain. Seek a licensed clinician for persistent, painful, "
        "rapidly changing, or concerning symptoms."
    )
    st.markdown(
        "**For more comparable check-ins:** use the same camera, distance, neutral "
        "expression, and soft consistent lighting; remove filters; and compare at "
        "regular intervals. A changing score alone does not establish improvement."
    )