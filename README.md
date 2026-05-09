# Three-Factor Authentication with Voice Recognition

This project is a Python Flask web app that demonstrates a three-factor authentication flow. A user logs in with a password, verifies a one-time password, and then completes voice verification using SpeechBrain speaker recognition.

Flask-based three-factor authentication demo using:

- Password login
- Email OTP
- Voice recognition with SpeechBrain

## How to Download and Run

Clone or download this repository, then open PowerShell inside the project folder.

Install everything required:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_INSTALL_FIRST.ps1
```

Start the Flask app:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_RUN_APP.ps1
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

## Run on Windows

Open PowerShell in this project folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_INSTALL_FIRST.ps1
```

Then start the app:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\MUST_RUN_APP.ps1
```

Open:

```text
http://127.0.0.1:5000
```

## Local OTP

The project defaults to local development mode:

```text
ENABLE_EMAIL_DELIVERY=false
```

In this mode, OTP codes are shown on the OTP page so you can test without Gmail SMTP credentials.

To send real email OTPs, copy/edit `email.env` and set:

```text
ENABLE_EMAIL_DELIVERY=true
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-gmail-app-password
```

Use a Gmail app password, not your normal Google password.

## Notes

- The first voice training or verification can take longer because the SpeechBrain model downloads into `pretrained_models/`.
- Local files such as `.venv`, `.python`, `.tools`, `email.env`, logs, local SQLite databases, and downloaded model files are intentionally ignored by Git.
