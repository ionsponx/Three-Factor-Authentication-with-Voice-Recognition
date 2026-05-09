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

## Real Gmail OTP Setup

For local testing, you do not need to configure Gmail. The OTP appears on the OTP page.

To send real OTP emails through your own Gmail account:

1. Run `MUST_INSTALL_FIRST.ps1` once. It creates a local `email.env` file from `email.env.example`.
2. Turn on 2-Step Verification in your Google Account.
3. Open Google's app-password page: [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
4. Create an app password for this project.
5. Copy the 16-character app password. Remove spaces if Google displays it in groups.
6. Edit `email.env`:

```text
ENABLE_EMAIL_DELIVERY=true
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-16-digit-google-app-password
```

Use a Google app password, not your normal Google password. Google app passwords require 2-Step Verification. See Google's official help page: [Sign in with app passwords](https://support.google.com/mail/answer/185833).

Do not upload `email.env` to GitHub. It is ignored by `.gitignore` because it contains private credentials.

## Voice Data and Database Files

The app stores local user and voice data in this SQLite database file:

```text
instance/site.db
```

This file is created automatically when the app runs. It is ignored by Git, so every user has their own local database.

After starting the app, open the local admin page to see registered users and stored voice recordings:

```text
http://127.0.0.1:5000/admin/users
```

On that page:

- `Voice Trained` shows whether a user has saved voice authentication data.
- `Stored Voice Recordings` links to that user's saved training recordings.
- `Delete` removes the user. The app is configured to delete that user's stored voice recordings at the same time.

To reset everything locally, stop the Flask app and delete:

```text
instance/site.db
```

The next app start will create a fresh empty database.

## Notes

- The first voice training or verification can take longer because the SpeechBrain model downloads into `pretrained_models/`.
- Local files such as `.venv`, `.python`, `.tools`, `email.env`, logs, local SQLite databases, and downloaded model files are intentionally ignored by Git.
