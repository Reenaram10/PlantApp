from app import app, db, text
with app.app_context():
    try:
        db.session.execute(text("ALTER TABLE orders ADD COLUMN amount_paid FLOAT DEFAULT 0.0;"))
        db.session.commit()
    except Exception as e:
        print(f"Error amount_paid: {e}")
    try:
        db.session.execute(text("ALTER TABLE orders ADD COLUMN balance FLOAT DEFAULT 0;"))
        # Using UPDATE to calculate initial balances. Prices and quantities can be null, handled safely here.
        db.session.execute(text("UPDATE orders SET balance = (COALESCE(price, 0) * COALESCE(quantity, 1)) WHERE balance = 0;"))
        db.session.commit()
        print("Completed")
    except Exception as e:
        print(f"Error balance: {e}")
