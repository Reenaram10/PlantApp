import os
import sys

sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = r'd:\PlantApp\temp\.keras'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=" * 70)
print("TESTING IMPORTS IN LEGACY CONTEXT:")
print("=" * 70)
try:
    import app_legacy
    print(" [1] app_legacy imported successfully.")
    
    from app.tools.plant_identifier import identify_plant
    print(" [2] from app.tools.plant_identifier import identify_plant: SUCCESS")
    print("     identify_plant function:", identify_plant)
    
    from app.history.history_retriever import history_manager
    print(" [3] from app.history.history_retriever import history_manager: SUCCESS")
    print("     history_manager object:", history_manager)
    print("     sys.modules['app']:", sys.modules.get('app'))
    print("     sys.modules['app_legacy']:", sys.modules.get('app_legacy'))
except Exception as e:
    print(" [!] Legacy import error:", e)

print("\n" + "=" * 70)
print("TESTING IMPORTS IN NEW CONTEXT (create_app / app.py):")
print("=" * 70)
try:
    from create_app import create_app
    app_instance = create_app()
    print(" [1] create_app() instantiated successfully.")
    
    from app.tools.plant_identifier import identify_plant as new_identify_plant
    print(" [2] from app.tools.plant_identifier import identify_plant: SUCCESS")
    print("     identify_plant function:", new_identify_plant)
    
    from app.history.history_retriever import history_manager as new_history_manager
    print(" [3] from app.history.history_retriever import history_manager: SUCCESS")
    print("     history_manager object:", new_history_manager)
    print("     sys.modules['app']:", sys.modules.get('app'))
except Exception as e:
    print(" [!] New import error:", e)

print("=" * 70)
