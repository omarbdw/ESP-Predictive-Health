"""Initial root-cause taxonomy for ESP health cases."""

FAILURE_CLASSES: dict[str, str] = {
    "NORMAL": "Normal ESP Operation",
    "GAS": "Gas Interference / Gas Lock",
    "LOW_INFLOW": "Pump-Off / Insufficient Inflow",
    "INORG_PLUG": "Scale / Inorganic Plugging",
    "ORG_PLUG": "Wax / Paraffin / Asphaltene Plugging",
    "SOLIDS": "Sand / Solids / Fines",
    "TUBING_LEAK": "Tubing Leak / Hole in Tubing",
    "PUMP_WEAR": "Pump Hydraulic Degradation / Wear",
    "OVERLOAD": "Mechanical Restriction / Overload",
    "ELECTRICAL": "Motor / Cable / Electrical Failure",
    "SURFACE": "VSD / Transformer / Surface Power Failure",
    "UNKNOWN": "Unknown / Unconfirmed",
}

WELL_TYPES: tuple[str, ...] = (
    "Oil Producer",
    "Water Producer",
    "Water Injector",
    "Other",
)

CONFIRMATION_LEVELS: tuple[str, ...] = (
    "A = Confirmed",
    "B = Highly Probable",
    "C = Suspected",
    "D = Unknown",
)
