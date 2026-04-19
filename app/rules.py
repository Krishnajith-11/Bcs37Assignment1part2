from datetime import datetime, timedelta

def calculate_risk(customer):
    complaints = sum(1 for t in customer["tickets"] if t["type"] == "complaint")
    charge_diff = customer["monthly_charges"] - customer["previous_month_charges"]
    contract = customer["contract_type"]

    # 🔥 HIGH RISK CONDITIONS
    if complaints >= 5:
        return "HIGH"

    if contract == "Month-to-Month" and complaints >= 3:
        return "HIGH"

    if charge_diff > 20:
        return "HIGH"

    # MEDIUM
    if complaints >= 2:
        return "MEDIUM"

    return "LOW"