import os
import re
import json
import time
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app, Response
from sqlalchemy import text
from extensions import db, langfuse_client
from models.chat import UserQuery, Query, QueryResult
from models.tenant import User
from auth.decorators import get_current_tenant
from services.chat.chat_service import process_message
from services.chat.dialog_state import USER_DIALOG_STATE
from services.llm_service import normalize_query
from services.speech_service import generate_speech
from services.rag_service import history_faiss
from services.observability import store_query

chat_bp = Blueprint('chat', __name__)

def extract_json(text):
    """
    Robustly extract JSON from text even if it contains extra text or markdown blocks.
    """
    if not text:
        return None
        
    # Try to find JSON block {...}
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        # Fallback: try raw json if no braces found (unlikely for objects)
        return json.loads(text)
    except Exception as e:
        pass

@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat API requests"""
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "")
        location = data.get("location", "")
        user_id = data.get("user_id")

        if not user_message:
            return jsonify({"status": "error", "message": "No message provided"}), 400
        history = data.get("history", [])

        response = process_message(user_message, location, user_id, history)
        
        # Save History to FAISS Engine
        if response and response.get("status") == "success" and response.get("reply"):
            history_faiss.add_interaction(user_id, user_message, response.get("reply"))
        
        if response.get("type") == "text":
            speech_response = generate_speech(response["reply"])
            if speech_response.get("status") == "success":
                response["audio"] = speech_response.get("audio")

        return jsonify(response)

    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({
            "status": "error",
            "type": "text",
            "reply": "Sorry, I encountered an error processing your request."
        }), 500

@chat_bp.route("/api/user_queries", methods=["POST"])
def save_user_query():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    plant_id = data.get("plant_id")  # optional
    query = data.get("query")

    if not user_id or not query:
        return jsonify({"status": "error", "message": "user_id and query required"}), 400

    try:
        new_q = UserQuery(
            user_id=user_id,
            plant_id=plant_id,
            query_text=query
        )
        db.session.add(new_q)
        db.session.commit()

        return jsonify({
            "status": "success",
            "message": "Query saved ",
            "uq_id": new_q.uq_id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@chat_bp.route('/api/user_queries/<user_id>', methods=['GET'])
def get_user_history(user_id):
    try:
        queries = UserQuery.query.filter_by(user_id=int(user_id) if str(user_id).isdigit() else 0).order_by(UserQuery.created_at.desc()).limit(10).all()
        return jsonify({
            "status": "success",
            "history": [{"query": q.query_text, "date": q.created_at.isoformat()} for q in queries]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@chat_bp.route("/api/history/<user_id>", methods=["GET"])
def get_history(user_id):
    rows = db.session.execute(text("""
        SELECT TOP 50 query_id, description, created_at
        FROM query
        WHERE user_id = :uid
        ORDER BY created_at DESC
    """), {"uid": user_id}).fetchall()

    history = []
    for r in rows:
        history.append({
            "query_id": r[0],
            "query": r[1],
            "time": str(r[2])
        })

    return jsonify({"status": "success", "history": history})

@chat_bp.route('/api/queries', methods=['GET', 'POST'])
def manage_queries():
    """Get or store queries"""
    try:
        if request.method == "GET":
            user_id = request.args.get("user_id")
            if user_id:
                queries = Query.query.filter_by(user_id=user_id).all()
            else:
                queries = Query.query.all()
            
            return jsonify({
                "status": "success",
                "queries": [{
                    "query_id": q.query_id,
                    "description": q.description,
                    "user_id": q.user_id,
                    "created_at": q.created_at.strftime("%Y-%m-%d %H:%M:%S")
                } for q in queries]
            })

        elif request.method == "POST":
            data = request.get_json() or {}
            description = data.get("description")
            user_id = data.get("user_id")  # Can be None
            
            if not description:
                return jsonify({"status": "error", "message": "Description required"}), 400
            
            query_id = store_query(description, user_id)
            return jsonify({
                "status": "success",
                "message": "Query stored",
                "query_id": query_id
            })

    except Exception as e:
        print(f"Queries API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@chat_bp.route('/api/history/query', methods=['POST'])
def flask_history_query():
    """
    Answers questions about user's past plant history using LangChain & FAISS.
    """
    plant_name = None
    try:
        from app.history.history_retriever import answer_from_history
        data = request.get_json() or {}
        query = data.get("query")
        plant_name = data.get("plant_name")

        if not query:
            return jsonify({"status": "error", "message": "query is required"}), 400

        result = answer_from_history(query=query, plant_name=plant_name)
        return jsonify(result)
    except Exception as e:
        print(f" [History API] Error processing query: {e}")
        return jsonify({"type": "history", "plant": plant_name or "Unknown", "answer": "I couldn't find that information in the stored history.", "sources": []}), 500

@chat_bp.route('/api/history/profile', methods=['POST'])
def flask_history_profile():
    """
    Adds a new plant profile history version if constant parameters changed.
    """
    try:
        from app.history.history_retriever import history_manager
        data = request.get_json() or {}
        if not data.get("plant_name"):
            return jsonify({"status": "error", "message": "plant_name is required"}), 400
        
        res = history_manager.add_plant_profile_version(data)
        return jsonify({"status": "success", "profile": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@chat_bp.route('/api/history/variable', methods=['POST'])
def flask_history_variable():
    """
    Updates current variable metrics without creating a profile history version.
    """
    try:
        from app.history.history_retriever import history_manager
        data = request.get_json() or {}
        p_name = data.get("plant_name")
        if not p_name:
            return jsonify({"status": "error", "message": "plant_name is required"}), 400

        updates = {k: v for k, v in data.items() if k != "plant_name" and v is not None}
        res = history_manager.update_variable_state(p_name, updates)
        return jsonify({"status": "success", "variables": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@chat_bp.route('/api/history/analysis', methods=['POST'])
def flask_history_analysis():
    """
    Records an image analysis event in history FAISS vector store.
    """
    try:
        from app.history.history_retriever import history_manager
        data = request.get_json() or {}
        p_name = data.get("plant_name")
        d_name = data.get("disease_name")
        if not p_name or not d_name:
            return jsonify({"status": "error", "message": "plant_name and disease_name are required"}), 400

        res = history_manager.add_analysis_history(
            plant_name=p_name,
            disease_name=d_name,
            plant_confidence=float(data.get("plant_confidence", 1.0)),
            disease_confidence=float(data.get("disease_confidence", 1.0)),
            timestamp=data.get("timestamp")
        )
        return jsonify({"status": "success", "analysis": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@chat_bp.route('/api/query_results', methods=['GET', 'POST'])
def manage_query_results():
    """Get or store query results"""
    try:
        if request.method == "GET":
            query_id = request.args.get("query_id")
            tenant_id = get_current_tenant()
            if query_id:
                results = QueryResult.query.filter_by(query_id=query_id, business_id=tenant_id).all()
            else:
                results = QueryResult.query.filter_by(business_id=tenant_id).all()
            
            return jsonify({
                "status": "success",
                "results": [{
                    "result_id": r.result_id,
                    "query_id": r.query_id,
                    "result_description": r.result_description,
                    "result_source": r.result_source,
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S")
                } for r in results]
            })

        elif request.method == "POST":
            data = request.get_json() or {}
            query_id = data.get("query_id")
            result_description = data.get("result_description")
            result_source = data.get("result_source")  # 'llm' or 'local'
            
            if not query_id or not result_description or not result_source:
                return jsonify({"status": "error", "message": "query_id, result_description, and result_source required"}), 400
            
            if result_source not in ['llm', 'local']:
                return jsonify({"status": "error", "message": "result_source must be 'llm' or 'local'"}), 400
            
            tenant_id = get_current_tenant()
            new_res = QueryResult(
                business_id=tenant_id,
                query_id=query_id,
                result_description=result_description,
                result_source=result_source
            )
            db.session.add(new_res)
            db.session.commit()
            return jsonify({
                "status": "success",
                "message": "Query result stored",
                "result_id": new_res.result_id
            })

    except Exception as e:
        print(f"Query Results API error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

