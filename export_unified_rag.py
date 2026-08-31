from app import app, db, Plant, Category
import json
import os

def export_unified_knowledge():
    with app.app_context():
        knowledge_base = []
        
        # 1. Plants
        plants = Plant.query.all()
        for p in plants:
            content = f"Plant Name: {p.plant_name}\nDescription: {p.description or 'No description available.'}"
            knowledge_base.append({
                "id": f"plant_{p.plant_id}",
                "type": "plant",
                "category": "Plant Information",
                "content": content,
                "target_id": p.plant_id,
                "plant_name": p.plant_name
            })
            
        # 2. Categories
        categories = Category.query.all()
        for c in categories:
            content = f"Category Name: {c.category_name}. This includes plants that belong to the {c.category_name} group."
            knowledge_base.append({
                "id": f"category_{c.category_id}",
                "type": "category",
                "category": "Plant Category",
                "content": content,
                "target_id": c.category_id,
                "category_name": c.category_name
            })
            
        # 3. Tomato Specialist Care (Already local)
        tomato_path = "tomato_profile.json"
        if os.path.exists(tomato_path):
            with open(tomato_path, 'r') as f:
                tomato_data = json.load(f)
                for item in tomato_data:
                    item["type"] = "care"
                    # Add plant name prefix to improve RAG matching
                    item["content"] = f"Tomato Specialist Care - {item['category']}: {item['content']}"
                    knowledge_base.append(item)
                
        output_path = "unified_plants_knowledge.json"
        with open(output_path, 'w') as f:
            json.dump(knowledge_base, f, indent=4)
            
        print(f"Exported {len(knowledge_base)} unified knowledge chunks to {output_path}")

if __name__ == "__main__":
    export_unified_knowledge()
