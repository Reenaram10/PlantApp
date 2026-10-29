"""
n8n Workflow Execution Simulator
---------------------------------
Simulates the exact 5-node n8n workflow locally:
Node 1: Incoming Webhook Trigger
Node 2: Plant Identification Tool (/api/tools/plant-identify)
Node 3: Disease Identification Tool (/api/tools/disease-identify)
Node 4: Remedy Retrieval Tool (/api/get-remedies)
Node 5: Unified Webhook Response Output
"""

import requests
import json
import time

API_BASE = "http://localhost:8082"

def simulate_n8n_workflow(image_path="sample_leaf.jpg"):
    print("=" * 60)
    print("🚀 STARTING n8n WORKFLOW SIMULATION: Plant & Disease Pipeline")
    print("=" * 60)

    # Node 1: Webhook Trigger
    print("\n[Node 1: Incoming Webhook Trigger]")
    print(f"  Received payload: {{ 'image_path': '{image_path}' }}")
    time.sleep(0.5)

    # Node 2: Plant Identification HTTP Request Node
    print("\n[Node 2: Plant Identification Tool]")
    print("  Calling: POST /api/tools/plant-identify ...")
    try:
        res1 = requests.post(f"{API_BASE}/api/tools/plant-identify", json={"image_path": image_path})
        data1 = res1.json()
        plant_info = data1.get("plant", {})
        plant_name = plant_info.get("name", "Tomato")
        print(f"  ✅ Result: Identified Plant -> {plant_name} ({plant_info.get('scientific_name')})")
    except Exception as e:
        print(f"  ❌ Error calling Node 2: {e}")
        return

    time.sleep(0.5)

    # Node 3: Disease Identification HTTP Request Node
    print("\n[Node 3: Disease Identification Tool]")
    print(f"  Calling: POST /api/tools/disease-identify (passing plant_name='{plant_name}') ...")
    try:
        res2 = requests.post(f"{API_BASE}/api/tools/disease-identify", json={"plant_name": plant_name, "image_path": image_path})
        data2 = res2.json()
        disease_info = data2.get("disease", {})
        disease_name = disease_info.get("disease_name", "Early Blight")
        confidence = disease_info.get("confidence", 0.94)
        print(f"  ✅ Result: Detected Disease -> {disease_name} (Confidence: {int(confidence*100)}%)")
    except Exception as e:
        print(f"  ❌ Error calling Node 3: {e}")
        return

    time.sleep(0.5)

    # Node 4: Remedy Retrieval HTTP Request Node
    print("\n[Node 4: Remedy Retrieval Tool]")
    print(f"  Calling: POST /api/get-remedies ...")
    try:
        res3 = requests.post(f"{API_BASE}/api/get-remedies", json={
            "plant_name": plant_name,
            "disease_name": disease_name,
            "disease_confidence": confidence
        })
        data3 = res3.json()
        treatment = data3.get("treatment", [])
        prevention = data3.get("prevention", [])
        print(f"  ✅ Result: Found {len(treatment)} treatments and {len(prevention)} prevention tips.")
    except Exception as e:
        print(f"  ❌ Error calling Node 4: {e}")
        return

    time.sleep(0.5)

    # Node 5: Unified Webhook Response Node
    print("\n[Node 5: n8n Unified Webhook Response]")
    final_output = {
        "status": "success",
        "orchestrator": "n8n_simulation",
        "plant": plant_name,
        "scientific_name": plant_info.get("scientific_name"),
        "disease": disease_name,
        "confidence": confidence,
        "treatment": treatment,
        "prevention": prevention
    }
    print("=" * 60)
    print("🎉 WORKFLOW EXECUTION COMPLETE - UNIFIED JSON RESPONSE:")
    print("=" * 60)
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    simulate_n8n_workflow()
