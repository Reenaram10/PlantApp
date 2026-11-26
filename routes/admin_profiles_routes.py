import os
import io
import json
import traceback
from datetime import datetime
import pandas as pd
from flask import Blueprint, request, jsonify, current_app, send_file, Response, make_response
from werkzeug.utils import secure_filename
from config import APP_ROOT, PROFILE_DATASETS
from models.catalog import PlantKnowledge, DiseaseKnowledge
from services.rag_service import faiss_rag

admin_profiles_bp = Blueprint('admin_profiles', __name__)

@admin_profiles_bp.route('/api/rebuild-index', methods=['POST'])
@admin_profiles_bp.route('/api/admin/rag/sync', methods=['POST'])
def sync_rag_index():
    """Force sync/rebuild FAISS RAG index from database and CSV files"""
    try:
        print(" [FAISS] Manual RAG index rebuild requested.")
        faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
        return jsonify({
            "status": "success", 
            "message": "FAISS vector index rebuilt successfully from database and CSV files."
        })
    except Exception as e:
        print(f" [FAISS] RAG rebuild error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@admin_profiles_bp.route('/api/admin/profile-stats', methods=['GET'])
def get_profile_stats():
    """
    Returns record counts, columns, file details, and preview rows for dataset profiles.
    """
    try:
        profile_type = request.args.get('profile_type', 'all')
        base_dir = APP_ROOT
        results = {}

        targets = [profile_type] if profile_type in PROFILE_DATASETS else list(PROFILE_DATASETS.keys())

        import pandas as pd
        for p_key in targets:
            info = PROFILE_DATASETS[p_key]
            file_path = os.path.join(base_dir, str(info["file"]))
            
            if not os.path.exists(file_path):
                results[p_key] = {
                    "key": p_key,
                    "title": info["title"],
                    "exists": False,
                    "row_count": 0,
                    "columns": info["columns"],
                    "preview": [],
                    "last_modified": "N/A"
                }
                continue

            # Load CSV safely with fallback encodings
            df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except Exception:
                    continue
            
            if df is None:
                df = pd.DataFrame()

            # Clean dataframe for JSON response (replace NaN with empty string)
            df_clean = df.fillna("")
            preview_rows = df_clean.head(10).to_dict(orient="records")
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

            results[p_key] = {
                "key": p_key,
                "title": info["title"],
                "exists": True,
                "row_count": len(df),
                "columns": list(df.columns),
                "required_cols": info["required_cols"],
                "preview": preview_rows,
                "last_modified": mod_time,
                "filename": info["file"]
            }

        return jsonify({"status": "success", "datasets": results})

    except Exception as e:
        print(f" [Profile Stats Error]: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@admin_profiles_bp.route('/api/admin/upload-profiles', methods=['POST'])
def upload_profile_dataset():
    """
    Accepts Excel (.xlsx / .xls) or CSV file uploads to update Geography, Plant, or Disease dataset profiles.
    Supports 'replace' or 'append' mode. Rebuilds FAISS RAG index automatically.
    """
    import io
    import pandas as pd
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded in form payload ('file' key expected)."}), 400

        file_obj = request.files['file']
        profile_type = request.form.get('profile_type', 'plant_profile')
        mode = request.form.get('mode', 'replace').lower() # 'replace' or 'append'

        if profile_type not in PROFILE_DATASETS:
            return jsonify({
                "status": "error",
                "message": f"Invalid profile_type. Must be one of: {list(PROFILE_DATASETS.keys())}"
            }), 400

        meta = PROFILE_DATASETS[profile_type]
        base_dir = APP_ROOT
        target_file_path = os.path.join(base_dir, str(meta["file"]))

        filename = (file_obj.filename or "").lower()
        content = file_obj.read()

        if not content:
            return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

        # Parse file based on extension
        new_df = None
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            new_df = pd.read_excel(io.BytesIO(content))
        else:
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    new_df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    break
                except Exception:
                    continue

        if new_df is None or new_df.empty:
            return jsonify({"status": "error", "message": "Could not parse spreadsheet data or sheet is empty."}), 400

        # Clean column names (strip whitespace)
        new_df.columns = [str(c).strip() for c in new_df.columns]

        # Check required columns
        missing_cols = [col for col in meta["required_cols"] if col not in new_df.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Uploaded file is missing required columns for {meta['title']}: {missing_cols}. Found columns: {list(new_df.columns)}"
            }), 400

        # Process dataset depending on mode
        final_df = None
        if mode == 'append' and os.path.exists(target_file_path):
            existing_df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    existing_df = pd.read_csv(target_file_path, encoding=enc)
                    break
                except Exception:
                    continue
            
            if existing_df is not None and not existing_df.empty:
                existing_df.columns = [str(c).strip() for c in existing_df.columns]
                # Combine dataframes
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                # Drop exact duplicate rows
                key_col = str(meta.get("key_col") or "")
                if key_col and key_col in combined.columns:
                    final_df = combined.drop_duplicates(subset=[key_col], keep="last")
                else:
                    final_df = combined.drop_duplicates(keep="last")
            else:
                final_df = new_df
        else:
            final_df = new_df

        # Fill NaNs with empty string
        final_df = final_df.fillna("")

        # Save back to target CSV file using latin1 or utf8
        try:
            final_df.to_csv(target_file_path, index=False, encoding='utf-8')
        except Exception:
            final_df.to_csv(target_file_path, index=False, encoding='latin1')

        # Trigger RAG FAISS re-indexing
        print(f" [Admin Upload] Syncing RAG vector database after {meta['title']} update...")
        rag_reindexed = False
        try:
            faiss_rag.sync_from_db(PlantKnowledge, DiseaseKnowledge)
            rag_reindexed = True
        except Exception as rag_err:
            print(f" [Admin Upload] FAISS RAG sync warning: {rag_err}")

        return jsonify({
            "status": "success",
            "message": f"Successfully updated {meta['title']} ({mode} mode). Total rows: {len(final_df)}.",
            "profile_type": profile_type,
            "mode": mode,
            "new_rows_uploaded": len(new_df),
            "total_rows": len(final_df),
            "rag_reindexed": rag_reindexed,
            "filename": meta["file"]
        })

    except Exception as e:
        print(f" [Upload Profile Error]: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error processing upload: {str(e)}"}), 500

@admin_profiles_bp.route('/api/admin/download-template/<profile_type>', methods=['GET'])
def download_profile_template(profile_type):
    """
    Generates and returns a downloadable CSV template with standard headers and sample data for admins.
    """
    try:
        if profile_type not in PROFILE_DATASETS:
            return jsonify({"status": "error", "message": "Invalid profile type."}), 400

        meta = PROFILE_DATASETS[profile_type]
        cols = list(meta["columns"]) if isinstance(meta["columns"], (list, tuple)) else []
        example = dict(meta["example"]) if isinstance(meta["example"], dict) else {}

        import pandas as pd
        template_df = pd.DataFrame([example], columns=cols)
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        response = make_response(csv_data)
        response.headers["Content-Disposition"] = f"attachment; filename={profile_type}_template.csv"
        response.headers["Content-type"] = "text/csv; charset=utf-8"
        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

