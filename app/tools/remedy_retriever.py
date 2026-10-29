def get_remedies(plant_name: str, disease_name: str, disease_confidence: float = 1.0, status: str = "detected") -> dict:
    """
    Retrieves treatment, prevention, and care tips for a plant and disease.
    
    IMPORTANT RULE:
    - Do NOT retrieve disease-specific remedies when disease confidence is below 90% (0.90) or status is 'uncertain'.
    
    Args:
        plant_name: Name of the plant (e.g. "Tomato").
        disease_name: Name of the disease (e.g. "Early Blight").
        disease_confidence: Confidence score of disease identification.
        status: Disease identification status ('detected', 'healthy', 'uncertain').
        
    Returns:
        dict with treatment, prevention, and care_tips lists.
    """
    # If disease confidence is below 90% or status is uncertain, DO NOT retrieve remedies
    if disease_confidence < 0.90 or status == "uncertain" or disease_name == "Unknown Disease":
        return {
            "treatment": [],
            "prevention": [],
            "care_tips": []
        }

    # If leaf is healthy
    if status == "healthy" or disease_name == "None Detected":
        return {
            "treatment": [],
            "prevention": [
                f"Maintain regular watering schedule suited for {plant_name}.",
                "Ensure proper soil drainage and adequate sunlight exposure.",
                "Inspect leaves periodically for early signs of pests or fungal spots."
            ],
            "care_tips": [
                f"Apply organic fertilizer during growing season for {plant_name}.",
                "Mulch around the plant base to conserve soil moisture.",
                "Prune dead or yellowing foliage to promote healthy circulation."
            ]
        }

    # Disease remedies dictionary lookup
    remedy_kb = {
        ("Tomato", "Early Blight"): {
            "treatment": [
                "Prune infected lower leaves showing target-like brown spots immediately.",
                "Apply copper-based fungicide or neem oil spray every 7-10 days."
            ],
            "prevention": [
                "Rotate crops annually and avoid planting nightshades in the same plot.",
                "Water at the base of the plant to keep foliage dry.",
                "Ensure proper spacing (24-36 inches) for airflow."
            ],
            "care_tips": [
                "Stake or cage tomato plants to keep fruit and leaves off the soil.",
                "Apply organic straw mulch to prevent soil-borne spore splash."
            ]
        },
        ("Potato", "Early Blight"): {
            "treatment": [
                "Apply bio-fungicide or approved copper spray at first symptom display.",
                "Remove and burn or destroy severely blighted vines."
            ],
            "prevention": [
                "Plant certified disease-free seed potatoes.",
                "Maintain optimal nitrogen levels to avoid plant stress."
            ],
            "care_tips": [
                "Hill soil around plants to protect developing tubers from spores."
            ]
        },
        ("Potato", "Late Blight"): {
            "treatment": [
                "Destroy infected plants immediately to prevent rapid spore spread.",
                "Apply systemic copper fungicide under severe conditions."
            ],
            "prevention": [
                "Plant resistant varieties and monitor weather for cool, humid conditions.",
                "Destroy volunteer potato plants in early spring."
            ],
            "care_tips": [
                "Ensure proper field drainage and avoid overhead irrigation."
            ]
        }
    }

    key = (plant_name, disease_name)
    if key in remedy_kb:
        return remedy_kb[key]

    # General default remedies for recognized diseases
    return {
        "treatment": [
            f"Apply organic neem oil or suitable fungicide targeted for {disease_name}.",
            "Remove affected leaves to reduce inoculum spread."
        ],
        "prevention": [
            f"Use crop rotation and disease-free seeds/starts for {plant_name}.",
            "Avoid overhead watering to keep foliage dry."
        ],
        "care_tips": [
            "Monitor plants weekly for early detection of foliage changes."
        ]
    }
