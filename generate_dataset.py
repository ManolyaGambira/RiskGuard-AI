import random
import pandas as pd

random.seed(42)

rows = []

for i in range(1, 501):

    # Customer spending baseline
    historical_avg = random.randint(500, 5000)

    # Most transactions are normal
    is_fraud = 1 if random.random() < 0.20 else 0

    if is_fraud:
        # Fraudulent transactions are more likely to contain
        # multiple suspicious signals.
        amount_ratio = random.uniform(5, 25)
        amount = round(historical_avg * amount_ratio)

        device_changed = random.choices([0, 1], weights=[20, 80])[0]
        location_changed = random.choices([0, 1], weights=[30, 70])[0]
        unusual_hour = random.choices([0, 1], weights=[30, 70])[0]
        velocity_1h = random.randint(3, 8)
        payment_method_changed = random.choices(
            [0, 1], weights=[30, 70]
        )[0]

    else:
        # Legitimate transactions can occasionally be unusual.
        amount_ratio = random.uniform(0.5, 4)
        amount = round(historical_avg * amount_ratio)

        device_changed = random.choices(
            [0, 1], weights=[90, 10]
        )[0]

        location_changed = random.choices(
            [0, 1], weights=[92, 8]
        )[0]

        unusual_hour = random.choices(
            [0, 1], weights=[95, 5]
        )[0]

        velocity_1h = random.choices(
            [1, 2, 3], weights=[80, 15, 5]
        )[0]

        payment_method_changed = random.choices(
            [0, 1], weights=[85, 15]
        )[0]

    rows.append({
        "transaction_id": f"MLTXN{i:04d}",
        "amount": amount,
        "historical_avg": historical_avg,
        "amount_ratio": round(amount_ratio, 2),
        "device_changed": device_changed,
        "location_changed": location_changed,
        "unusual_hour": unusual_hour,
        "velocity_1h": velocity_1h,
        "payment_method_changed": payment_method_changed,
        "is_fraud": is_fraud
    })


df = pd.DataFrame(rows)

df.to_csv("data/ml_transactions.csv", index=False)

print("Dataset generated successfully!")
print("Shape:", df.shape)

print("\nFraud distribution:")
print(df["is_fraud"].value_counts())

print("\nFraud percentage:")
print(df["is_fraud"].mean() * 100)

print("\nFirst 5 rows:")
print(df.head())