import ast
import datetime
import glob
import os


def collect_functions(filepath):
    tree = ast.parse(open(filepath, encoding="utf-8").read(), filename=filepath)
    return [
        (node.name, node, filepath)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def extract_str_constants(stmts):
    """Collect all ast.Constant str values (including f-string literal parts), excluding leading docstrings."""
    strs = []
    for idx, stmt in enumerate(stmts):
        if idx == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strs.append(node.value)
    return strs


def main():
    legacy_funcs = collect_functions("app_legacy.py")
    new_files = (
        ["app.py", "create_app.py", "config.py", "extensions.py"]
        + sorted(glob.glob("auth/**/*.py", recursive=True))
        + sorted(glob.glob("models/**/*.py", recursive=True))
        + sorted(glob.glob("services/**/*.py", recursive=True))
        + sorted(glob.glob("routes/**/*.py", recursive=True))
    )

    new_func_map = {}
    for nf in new_files:
        if os.path.exists(nf):
            for name, node, path in collect_functions(nf):
                new_func_map.setdefault(name, []).append((node, path))

    chat_helpers = [
        "classify_query_intent", "classify_category", "check_db_plant_availability",
        "make_plant_unavailable_guardrail", "format_plant_profile", "fetch_db_plant_details",
        "_handle_dialog_state", "_handle_cart_display", "_handle_db_inventory_query",
        "_handle_buy_intent", "clean_text", "classify_affirmation",
    ]

    lines_out = []
    diffs_found = 0
    for name, leg_node, leg_path in legacy_funcs:
        matches = new_func_map.get(name, [])
        if not matches:
            continue
        new_node, new_path = matches[0]
        leg_strs = set(extract_str_constants(leg_node.body))
        new_strs = set(extract_str_constants(new_node.body))
        if name == "process_message":
            for h in chat_helpers:
                if h in new_func_map:
                    new_strs.update(extract_str_constants(new_func_map[h][0][0].body))

        only_leg = sorted(leg_strs - new_strs)
        only_new = sorted(new_strs - leg_strs)
        if only_leg or only_new:
            diffs_found += 1
            lines_out.append(f"[{name}] ({leg_path}:{leg_node.lineno} vs {new_path}:{new_node.lineno})")
            for s in only_leg:
                lines_out.append(f"  - LEGACY ONLY: {repr(s)}")
            for s in only_new:
                lines_out.append(f"  + NEW ONLY:    {repr(s)}")

    os.makedirs("scratch/logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"scratch/logs/string_compare_{ts}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"Functions with string constant differences: {diffs_found}")
    print(f"Saved to {out_path}")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
