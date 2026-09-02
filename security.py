"""
Account-security helpers layered on top of the original face+OTP system:

  - password hashing (3rd factor, alongside face + OTP)
  - recovery codes (one-time backup login if face/OTP/webcam is unavailable)
  - failed-attempt lockout (brute-force protection)
  - CSV attempt logging (audit trail)

Kept in its own module so main_server.py doesn't turn into one giant file.
"""

import os
import csv
import json
import secrets
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
LOG_FILE = os.path.join(BASE_DIR, "login_log.csv")

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 5
NUM_RECOVERY_CODES = 8


# ---------- users.json read/write ----------

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def new_user_record(secret, password):
    """Shape of one entry in users.json."""
    return {
        "secret": secret,                       # per-user TOTP secret
        "password_hash": generate_password_hash(password),
        "recovery_codes": [],                    # filled in by generate_recovery_codes()
        "failed_attempts": 0,
        "locked_until": None,                     # ISO timestamp string, or None
        "last_login": None,                        # ISO timestamp string, or None
    }


# ---------- password (3rd factor) ----------

def check_password(user_record, password):
    return check_password_hash(user_record["password_hash"], password or "")


# ---------- recovery codes ----------

def generate_recovery_codes(users, username):
    """Create NUM_RECOVERY_CODES fresh one-time codes, store only their
    hashes (never the plaintext), and return the plaintext codes ONCE so
    the page can show them to the user to save somewhere safe."""
    plaintext_codes = [secrets.token_hex(4) for _ in range(NUM_RECOVERY_CODES)]
    users[username]["recovery_codes"] = [
        {"hash": generate_password_hash(code), "used": False} for code in plaintext_codes
    ]
    save_users(users)
    return plaintext_codes


def use_recovery_code(users, username, code):
    """Check a submitted recovery code against the stored hashes. If it
    matches an unused one, mark it used (one-time only) and return True."""
    if username not in users:
        return False
    for entry in users[username].get("recovery_codes", []):
        if not entry["used"] and check_password_hash(entry["hash"], code or ""):
            entry["used"] = True
            save_users(users)
            return True
    return False


# ---------- lockout ----------

def is_locked(user_record):
    locked_until = user_record.get("locked_until")
    if not locked_until:
        return False, None
    locked_until_dt = datetime.fromisoformat(locked_until)
    if datetime.now() < locked_until_dt:
        return True, locked_until_dt
    return False, None


def record_failed_attempt(users, username):
    user = users[username]
    user["failed_attempts"] = user.get("failed_attempts", 0) + 1
    if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
        user["locked_until"] = (datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
        user["failed_attempts"] = 0  # reset counter once locked
    save_users(users)


def record_successful_login(users, username):
    user = users[username]
    user["failed_attempts"] = 0
    user["locked_until"] = None
    user["last_login"] = datetime.now().isoformat()
    save_users(users)


# ---------- attempt logging (audit trail) ----------

def log_attempt(username, password_ok, otp_ok, face_ok, result):
    is_new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "username", "password_ok", "otp_ok", "face_ok", "result"])
        writer.writerow([datetime.now().isoformat(), username, password_ok, otp_ok, face_ok, result])
