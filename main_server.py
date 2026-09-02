from flask import Flask, render_template, request, session, redirect, url_for
import pyotp
import qrcode
import os
import json
from werkzeug.utils import secure_filename
from face_recognition import enroll_face, verify_face

app = Flask(__name__)

# Anchor every path to this file's own directory, not the current working
# directory the app happens to be launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")
QR_DIR = os.path.join(STATIC_DIR, "qrcodes")
SESSION_SECRET_FILE = os.path.join(BASE_DIR, "flask_secret.txt")

os.makedirs(QR_DIR, exist_ok=True)

# Persist Flask's session-signing key so existing sessions survive restarts
# (same reasoning as persisting the TOTP secret in the old single-user app).
if os.path.exists(SESSION_SECRET_FILE):
    with open(SESSION_SECRET_FILE) as f:
        app.secret_key = f.read().strip()
else:
    app.secret_key = os.urandom(24).hex()
    with open(SESSION_SECRET_FILE, "w") as f:
        f.write(app.secret_key)

ISSUER = "My2FAProject"


def load_users():
    """users.json maps username -> {"secret": "<per-user TOTP secret>"}."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def safe_username(raw):
    """Turn arbitrary input into a filesystem-safe username, or None if empty.
    This also blocks path-traversal tricks in filenames like face_models/<username>.yml."""
    name = secure_filename((raw or "").strip().lower())
    return name or None


@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = safe_username(request.form.get("username", ""))
        if not username:
            return render_template("signup.html", error="Enter a valid username.")

        users = load_users()
        if username in users:
            return render_template(
                "signup.html",
                error=f"'{username}' is already registered. Try logging in instead.",
            )

        # Each user gets their OWN TOTP secret - not one shared secret for
        # the whole app, so one person's code can never verify another's login.
        secret = pyotp.random_base32()
        users[username] = {"secret": secret}
        save_users(users)

        uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=ISSUER)
        qr_filename = f"{username}.png"
        qrcode.make(uri).save(os.path.join(QR_DIR, qr_filename))

        session["pending_user"] = username

        return render_template("signup.html", username=username, qr_filename=qr_filename)

    return render_template("signup.html")


@app.route("/enroll-face", methods=["GET", "POST"])
def enroll_face_route():
    username = safe_username(session.get("pending_user") or request.args.get("user", ""))
    if not username:
        return redirect(url_for("register"))

    if request.method == "POST":
        try:
            enroll_face(username)
            return render_template(
                "status.html",
                title="Face enrolled successfully!",
                icon="&#9989;",
                message=f"'{username}' can now log in with their face + their own OTP code.",
                link_href=url_for("verify"),
                link_text="Continue to Login",
            )
        except Exception as e:
            return render_template(
                "status.html",
                title="Enrollment failed",
                icon="&#10060;",
                message=str(e),
                link_href=url_for("enroll_face_route"),
                link_text="Try Again",
            )

    return render_template("face_setup.html", username=username)


@app.route("/verify", methods=["GET", "POST"])
def verify():
    if request.method == "POST":
        username = safe_username(request.form.get("username", ""))
        code = request.form.get("code")

        users = load_users()
        if not username or username not in users:
            return render_template(
                "status.html",
                title="Unknown user",
                icon="&#10060;",
                message="No account with that username. Register first.",
                link_href=url_for("register"),
                link_text="Register",
            )

        # This user's own secret - a valid code from anyone else's secret
        # will NOT pass here.
        totp = pyotp.TOTP(users[username]["secret"])

        try:
            # This user's own face model - matching another enrolled
            # person's face will NOT pass here.
            face_ok = verify_face(username)
        except Exception as e:
            return render_template(
                "status.html",
                title="Face verification error",
                icon="&#10060;",
                message=str(e),
                link_href=url_for("enroll_face_route", user=username),
                link_text="Enroll your face first",
            )

        otp_ok = totp.verify(code, valid_window=1) if code else False

        print(f"[{username}] face_ok={face_ok} otp_ok={otp_ok}")

        if face_ok and otp_ok:
            return render_template(
                "status.html",
                title="2FA Verification Successful!",
                icon="&#9989;",
                message=f"Welcome, {username}. Face matched and your OTP was correct.",
                link_href=url_for("home"),
                link_text="Back to home",
            )

        if not face_ok and not otp_ok:
            reason = "Face didn't match this account, and the OTP was incorrect."
        elif not face_ok:
            reason = f"The OTP was correct, but the face didn't match {username}'s enrolled face."
        else:
            reason = "Face matched, but the OTP was incorrect or expired."

        return render_template(
            "status.html",
            title="Verification Failed",
            icon="&#10060;",
            message=reason,
            link_href=url_for("verify"),
            link_text="Try Again",
        )

    return render_template("login.html")


if __name__ == "__main__":
    app.run(debug=False)
