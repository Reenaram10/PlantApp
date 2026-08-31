from app import app, db, Plant
import json
import os

def export_plants():
    with app.app_context():
        plants = Plant.query.all()
        print(f"Found {len(plants)} plants in database.")
        
        knowledge_base = []
        for p in plants:
            content = f"Plant Name: {p.plant_name}\nDescription: {p.description or 'No description available.'}"
            knowledge_base.append({
                "id": f"plant_{p.plant_id}",
                "category": "General Plant Info",
                "content": content,
                "plant_id": p.plant_id,
                "plant_name": p.plant_name
            })
            
        # Also add the tomato specialist knowledge if we want to combine them
        tomato_path = "tomato_profile.json"
        if os.path.exists(tomato_path):
            with open(tomato_path, 'r') as f:
                knowledge_base.extend(json.load(f))
                
        output_path = "general_plants_knowledge.json"
        with open(output_path, 'w') as f:
            json.dump(knowledge_base, f, indent=4)
            
        print(f"Exported {len(knowledge_base)} knowledge chunks to {output_path}")

if __name__ == "__main__":
    export_plants()
