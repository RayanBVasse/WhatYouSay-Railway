from flask import Flask, request, redirect, url_for, abort, render_template_string, render_template
from werkzeug.utils import secure_filename
from collections import Counter
import os
import re

# ------------------
# Config
# ------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


# ------------------
# Helpers
# ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_exists_in_text(handle, text):
    handle = handle.strip().lower()
    text = text.lower()

    # common WhatsApp patterns
    patterns = [
        f"{handle}:",
        f"] {handle}:",
        f"\n{handle}:",
    ]

    return any(p in text for p in patterns)

# ------------------
# Routes
# ------------------
@app.route("/WhatYouSay/", methods=["GET"])
def index():
    return app.send_static_file("index.html")


@app.route("/WhatYouSay/upload", methods=["POST"])
def upload():
    # 1. user handle
    user_handle = request.form.get("user_handle", "").strip()
    platform = request.form.get("platform", "")
    file = request.files.get("text_file")

    if not user_handle:
        abort(400, "Missing user handle")
    
    if platform not in ["android", "ios"]:
        abort(400, "Platform selection required")
    
    # 2. file
    if "text_file" not in request.files:
        abort(400, "No file part")

    file = request.files["text_file"]

    if file.filename == "":
        abort(400, "No selected file")

    if not allowed_file(file.filename):
        abort(400, "Invalid file type")

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # -----
    # TEMP: read content (later → analysis pipeline)
    # -----
    with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    char_count = len(text)
    line_count = len(text.splitlines())

    if not handle_exists_in_text(user_handle, text):
        return render_template(
        "error.html",
        message=f"The handle '{user_handle}' does not appear in the uploaded file."
    )

# --- data preparation (THIS is where your code goes) ---
    speaker_counts = extract_speakers(text, user_handle)
    total_messages = sum(speaker_counts.values())
    user_messages = speaker_counts.get("You", 0)
    
    warnings = []

# Individual analysis threshold
    if user_messages < 250:
        warnings.append(
            f"Individual-level analysis may be unreliable: "
            f"only {user_messages} messages found (minimum recommended: 250)."
        )

# Group comparative threshold
    user_share = (user_messages / total_messages) * 100 if total_messages > 0 else 0

    MIN_SHARE_PERCENT = 5  # adjust later if needed

    if user_share < MIN_SHARE_PERCENT:
        warnings.append(
            f"Group comparison may be unreliable: "
            f"your contribution is {user_share:.1f}% of the corpus "
            f"(minimum recommended: {MIN_SHARE_PERCENT}%)."
        )



    top_speakers = list(speaker_counts.items())[:5]

    # --- render confirmation page ---
    return render_template(
         "confirmation.html",
            user_handle=user_handle,
            platform=platform,
            char_count=len(text),
            line_count=len(text.splitlines()),
            message_count=total_messages,
            top_speakers=top_speakers,
            chart_labels=list(speaker_counts.keys()),
            chart_values=list(speaker_counts.values()),
            warnings=warnings,
            user_messages=user_messages,
            user_share=user_share
    )


def extract_speakers(text, user_handle):
    speakers = []

    # WhatsApp speaker pattern (very conservative)
    # Examples:
    # "12/03/25, 09:41 - Vanessa:"
    # "[12/03/25, 09:41] Vanessa:"
    pattern = re.compile(r"[-\]]\s([^:]{1,40}):")

    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            name = match.group(1).strip()
            speakers.append(name)

    counts = Counter(speakers)

    anonymized = {}
    anon_index = 1

    for speaker, count in counts.most_common():
        if speaker.lower() == user_handle.lower():
            anonymized["You"] = count
        else:
            anonymized[f"Participant {anon_index}"] = count
            anon_index += 1

    return anonymized
# ------------------
# Local run
# ------------------
if __name__ == "__main__":
    app.run(debug=True)
