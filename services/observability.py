"""
Observability and Telemetry Service
Provides trace logging to Langfuse for LLM, RAG, and classification telemetry.
"""
import os
from sqlalchemy import text
from extensions import db, langfuse_client, LANGFUSE_ENABLED
from models.chat import Query

def log_user_interest(user_id, plant_id, interest_type, notes=""):
    try:
        if not user_id or str(user_id) in ["anonymous", "guest_user"]:
            return

        db.session.execute(text("""
            INSERT INTO user_interest (user_id, plant_id, interest_type, notes)
            VALUES (:user_id, :plant_id, :interest_type, :notes)
        """), {
            "user_id": int(user_id),
            "plant_id": int(plant_id),
            "interest_type": interest_type,
            "notes": notes
        })

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        print(" user_interest insert failed:", e)

def log_category_interest(user_id, category_id, interest_type="search"):
    if not user_id or not str(user_id).isdigit():
        return

    try:
        if db.engine.dialect.name == 'mssql':
            db.session.execute(text("""
                IF EXISTS (SELECT 1 FROM user_category_interest WHERE user_id = :u AND category_id = :c AND interest_type = :t)
                    UPDATE user_category_interest 
                    SET count = count + 1, last_interaction = CURRENT_TIMESTAMP 
                    WHERE user_id = :u AND category_id = :c AND interest_type = :t
                ELSE
                    INSERT INTO user_category_interest (user_id, category_id, interest_type, count, last_interaction)
                    VALUES (:u, :c, :t, 1, CURRENT_TIMESTAMP)
            """), {
                "u": int(user_id),
                "c": int(category_id),
                "t": interest_type
            })
        else:
            db.session.execute(text("""
                INSERT INTO user_category_interest
                (user_id, category_id, interest_type, count)
                VALUES (:u, :c, :t, 1)
                ON CONFLICT (user_id, category_id, interest_type)
                DO UPDATE SET
                    count = user_category_interest.count + 1,
                    last_interaction = CURRENT_TIMESTAMP
            """), {
                "u": int(user_id),
                "c": int(category_id),
                "t": interest_type
            })

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(" Category interest failed:", e)

def store_query(description, user_id, canonical_key=None):
    """Store user query"""
    try:
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            print(f" Invalid user_id: {user_id}, using 1 as default")
            user_id_int = 1
        
        query = Query(
            description=description,
            canonical_key=canonical_key,
            user_id=user_id_int
        )
        
        db.session.add(query)
        db.session.commit()
        print(f" Stored query: {description}")
        return query.query_id
    except Exception as e:
        print(f" Error storing query: {e}")
        db.session.rollback()
        return None

def log_langfuse_trace(trace_name, user_input, output_reply, metadata=None, level="DEFAULT", tags=None):
    """Log a completed LLM/RAG trace with timing metrics & failure telemetry to Langfuse."""
    if not LANGFUSE_ENABLED or not langfuse_client:
        return
    try:
        meta = metadata or {}
        trace_tags = tags or (["failure", "error"] if level == "ERROR" else ["performance", "telemetry"])
        
        # Support both Langfuse SDK v2/v3 (client.trace) and v4 (client.start_observation)
        if hasattr(langfuse_client, "trace") and callable(getattr(langfuse_client, "trace")):
            langfuse_client.trace(
                name=trace_name,
                input=user_input,
                output=output_reply,
                metadata=meta,
                tags=trace_tags
            )
        elif hasattr(langfuse_client, "start_observation") and callable(getattr(langfuse_client, "start_observation")):
            obs = langfuse_client.start_observation(
                name=trace_name,
                input=user_input,
                output=output_reply,
                metadata={**meta, "tags": trace_tags},
                level=level if level in ("DEBUG", "DEFAULT", "WARNING", "ERROR") else "DEFAULT"
            )
            obs.end()
        langfuse_client.flush()
    except Exception as e:
        print(f" [Langfuse] Logging trace exception: {e}")
