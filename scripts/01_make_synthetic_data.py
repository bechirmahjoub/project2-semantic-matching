import random
import pandas as pd

random.seed(42)

EQUIPMENT = [
    ("Hydraulic Pump", "HP-200", "hydraulic pump for industrial press lines"),
    ("Pressure Sensor", "PS-10", "high precision pressure sensor for pipelines"),
    ("Thermal Camera", "TC-IR", "infrared thermal camera for inspection"),
    ("Bearing", "BR-6205", "deep groove ball bearing for motors"),
    ("Conveyor Belt", "CB-3M", "conveyor belt for assembly lines"),
    ("PLC Controller", "PLC-X1", "programmable logic controller for automation"),
    ("Flow Meter", "FM-100", "electromagnetic flow meter for liquids"),
    ("Valve Actuator", "VA-24", "electric valve actuator 24V"),
    ("Power Supply", "PSU-12", "12V stabilized power supply"),
    ("Laser Distance Sensor", "LDS-50", "laser distance sensor up to 50m"),
]

SYNONYMS = {
    "hydraulic pump": ["hydraulic pump", "oil pump", "hydro pump", "hydraulic pumping unit"],
    "pressure sensor": ["pressure sensor", "bar sensor", "pressure probe", "psi sensor"],
    "thermal camera": ["thermal camera", "infrared camera", "ir camera", "heat camera"],
    "bearing": ["bearing", "ball bearing", "motor bearing", "6205 bearing"],
    "conveyor belt": ["conveyor belt", "belt conveyor", "assembly belt", "transport belt"],
    "plc controller": ["plc", "plc controller", "automation controller", "programmable controller"],
    "flow meter": ["flow meter", "liquid meter", "flow sensor", "electromagnetic flow meter"],
    "valve actuator": ["valve actuator", "electric actuator", "actuated valve motor", "24v actuator"],
    "power supply": ["power supply", "psu", "12v supply", "adapter 12v"],
    "laser distance sensor": ["laser sensor", "distance sensor", "range finder", "laser range sensor"],
}

def make_typos(s: str) -> str:
    if len(s) < 5 or random.random() < 0.6:
        return s
    i = random.randint(1, len(s) - 2)
    return s[:i] + s[i+1] + s[i] + s[i+2:]  # swap 2 chars

def main():
    equipment_df = pd.DataFrame(
        [{"item_id": sku, "name": name, "description": desc} for name, sku, desc in EQUIPMENT]
    )
    equipment_df.to_csv("data/raw/equipment.csv", index=False)

    queries = []
    for name, sku, _ in EQUIPMENT:
        key = name.lower()
        syn_key = None
        for k in SYNONYMS:
            if k in key:
                syn_key = k
                break
        syn_key = syn_key or key

        for _ in range(25):
            q = random.choice(SYNONYMS.get(syn_key, [name]))
            q = make_typos(q)
            if random.random() < 0.4:
                q = f"need {q} urgent for maintenance"
            queries.append({"query": q, "true_item_id": sku})

    pd.DataFrame(queries).to_csv("data/raw/queries.csv", index=False)
    print("Generated data/raw/equipment.csv and data/raw/queries.csv")

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    main()
