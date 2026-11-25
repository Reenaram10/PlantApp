from flask import g
from extensions import db
from models.tenant import AuditLog

def log_audit_action(action, module, record_id=None, details=None, business_id=None, user_id=None, user_email=None):
    try:
        biz_id = business_id or (g.current_user.get('business_id') if hasattr(g, 'current_user') and g.current_user else 'NURS-0001')
        u_id = user_id or (g.current_user.get('user_id') if hasattr(g, 'current_user') and g.current_user else None)
        u_email = user_email or (g.current_user.get('email') if hasattr(g, 'current_user') and g.current_user else None)
        
        entry = AuditLog(
            business_id=biz_id,
            user_id=u_id,
            user_email=u_email,
            action=action,
            module=module,
            record_id=str(record_id) if record_id else None,
            details=details
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("Audit log failed:", e)
