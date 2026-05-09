# Run Locally

This project is a Flask app with SQLite, email OTP, and SpeechBrain voice recognition.

## First setup

Open PowerShell in this folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_INSTALL_FIRST.ps1
```

The setup script downloads `uv`, installs a local Python 3.11 runtime, creates `.venv`, installs dependencies, and creates `email.env`.

## Start the app

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_RUN_APP.ps1
```

Open:

```text
http://127.0.0.1:5000
```

## Local OTP behavior

`email.env` defaults to:

```text
ENABLE_EMAIL_DELIVERY=false
```

With that setting, OTPs are shown directly on the OTP page so you can test the app without Gmail credentials.

To send real email, set `ENABLE_EMAIL_DELIVERY=true` and fill in `SENDER_EMAIL` and `SENDER_PASSWORD` in `email.env`.

## Voice model behavior

The SpeechBrain speaker model loads only when you train or verify voice. The first voice action can take a while because the model is downloaded into `pretrained_models/`.
