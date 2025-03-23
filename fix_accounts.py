from main import app, db, User
import random

with app.app_context():
    users = User.query.all()
    for user in users:
        if user.account_number == 'TEMP1234567890':
            while True:
                new_account = f"#{random.randint(1000000000, 9999999999)}"
                if not User.query.filter_by(account_number=new_account).first():
                    user.account_number = new_account
                    break
    db.session.commit()