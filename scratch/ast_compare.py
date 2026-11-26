import ast
import copy
import datetime
import difflib
import glob
import os


KNOWN_ATTR_SWAPS = {
    ("app", "logger"): ("current_app", "logger"),
    ("app", "app_context"): ("current_app", "app_context"),
}

KNOWN_NAME_TO_ATTR = {
    "LLM_OPTION": ("runtime_settings", "llm_option"),
    "STT_OPTION": ("runtime_settings", "stt_option"),
    "TTS_OPTION": ("runtime_settings", "tts_option"),
    "LLM_FAILSAFE": ("runtime_settings", "llm_failsafe"),
    "STT_FAILSAFE": ("runtime_settings", "stt_failsafe"),
    "TTS_FAILSAFE": ("runtime_settings", "tts_failsafe"),
}


class BaseNormalizer(ast.NodeTransformer):
    """Normalize only known refactor substitutions."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            pair = (node.value.id, node.attr)
            if pair in KNOWN_ATTR_SWAPS:
                new_base, new_attr = KNOWN_ATTR_SWAPS[pair]
                return ast.copy_location(
                    ast.Attribute(
                        value=ast.Name(id=new_base, ctx=ast.Load()),
                        attr=new_attr,
                        ctx=node.ctx,
                    ),
                    node,
                )
        return node

    def visit_Name(self, node):
        if node.id in KNOWN_NAME_TO_ATTR and isinstance(node.ctx, ast.Load):
            base, attr = KNOWN_NAME_TO_ATTR[node.id]
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id=base, ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                ),
                node,
            )
        return node


class ReqJsonNormalizer(ast.NodeTransformer):
    """Normalize request.json -> request.get_json() or {} (and request.json or {} -> request.get_json() or {})."""

    def visit_BoolOp(self, node):
        if (
            isinstance(node.op, ast.Or)
            and len(node.values) == 2
            and isinstance(node.values[1], ast.Dict)
            and not node.values[1].keys
        ):
            v0 = node.values[0]
            if isinstance(v0, ast.Attribute) and isinstance(v0.value, ast.Name) and v0.value.id == "request" and v0.attr == "json":
                return ast.copy_location(
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.Call(
                                func=ast.Attribute(value=ast.Name(id="request", ctx=ast.Load()), attr="get_json", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            ),
                            ast.Dict(keys=[], values=[]),
                        ],
                    ),
                    node,
                )
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "request" and node.attr == "json":
            return ast.copy_location(
                ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.Call(
                            func=ast.Attribute(value=ast.Name(id="request", ctx=ast.Load()), attr="get_json", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        ast.Dict(keys=[], values=[]),
                    ],
                ),
                node,
            )
        return node


class AppRootNormalizer(ast.NodeTransformer):
    """Normalize os.path.dirname(os.path.abspath(__file__)) -> APP_ROOT and app.root_path -> current_app.root_path."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "app" and node.attr == "root_path":
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="current_app", ctx=ast.Load()), attr="root_path", ctx=node.ctx),
                node,
            )
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        s = ast.unparse(node)
        if s == "os.path.dirname(os.path.abspath(__file__))":
            return ast.copy_location(ast.Name(id="APP_ROOT", ctx=ast.Load()), node)
        return node


def strip_docstring(stmts):
    if stmts and isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant) and isinstance(stmts[0].value.value, str):
        return stmts[1:]
    return stmts


def normalize_stmts(stmts, extra_transformers=()):
    stmts = strip_docstring(stmts)
    out = []
    for stmt in stmts:
        c = copy.deepcopy(stmt)
        c = BaseNormalizer().visit(c)
        for tr in extra_transformers:
            c = tr().visit(c)
        ast.fix_missing_locations(c)
        out.append(c)
    return out


def dump_stmts(stmts):
    return "\n".join(ast.dump(s, indent=2) for s in stmts)


def unparse_stmts(stmts):
    return "\n".join(ast.unparse(s) for s in stmts).splitlines(keepends=True)


def collect_functions(filepath):
    tree = ast.parse(open(filepath, encoding="utf-8").read(), filename=filepath)
    return [
        (node.name, node, filepath)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


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

    identical_count = 0
    missing_legacy = []
    class_a = []  # only request.json -> request.get_json() or {}
    class_b = []  # only APP_ROOT / current_app
    class_c = []  # only runtime_settings (or global LLM_OPTION removal)
    class_d = []  # anything else

    chat_related_diffs = []

    for name, leg_node, leg_path in legacy_funcs:
        matches = new_func_map.get(name, [])
        if not matches:
            missing_legacy.append((name, leg_node.lineno))
            continue

        new_node, new_path = matches[0]
        leg_base = normalize_stmts(leg_node.body)
        new_base = normalize_stmts(new_node.body)

        if dump_stmts(leg_base) == dump_stmts(new_base):
            identical_count += 1
            continue

        # Classify remaining difference
        leg_a = normalize_stmts(leg_node.body, (ReqJsonNormalizer,))
        new_a = normalize_stmts(new_node.body, (ReqJsonNormalizer,))
        leg_b = normalize_stmts(leg_node.body, (AppRootNormalizer,))
        new_b = normalize_stmts(new_node.body, (AppRootNormalizer,))

        # Check (c): runtime_settings substitutions (e.g. global LLM_OPTION removed)
        leg_c = [s for s in leg_base if not isinstance(s, ast.Global)]
        new_c = [s for s in new_base if not isinstance(s, ast.Global)]

        diff = "".join(
            difflib.unified_diff(
                unparse_stmts(leg_base),
                unparse_stmts(new_base),
                fromfile=f"{leg_path}:{leg_node.lineno} ({name})",
                tofile=f"{new_path}:{new_node.lineno} ({name})",
            )
        ).rstrip()

        if new_path.startswith("services\\chat\\") or new_path.startswith("services/chat/"):
            chat_related_diffs.append((name, leg_path, leg_node.lineno, new_path, new_node.lineno, diff))

        if dump_stmts(leg_a) == dump_stmts(new_a):
            class_a.append(name)
        elif dump_stmts(leg_b) == dump_stmts(new_b):
            class_b.append(name)
        elif dump_stmts(leg_c) == dump_stmts(new_c):
            class_c.append(name)
        else:
            class_d.append((name, leg_path, leg_node.lineno, new_path, new_node.lineno, diff))

    lines_out = []
    lines_out.append(f"Total functions in app_legacy.py: {len(legacy_funcs)}")
    lines_out.append(f"Identical after docstring + known normalization: {identical_count}")
    lines_out.append(f"Missing in new modules: {len(missing_legacy)} -> {[m[0] for m in missing_legacy]}")
    lines_out.append(f"Remaining differing functions: {len(class_a) + len(class_b) + len(class_c) + len(class_d)}")
    lines_out.append("")
    lines_out.append(f"Class (a) Only request.json -> request.get_json() or {{}}: {len(class_a)}")
    lines_out.append(f"  {class_a}")
    lines_out.append(f"Class (b) Only APP_ROOT / current_app: {len(class_b)}")
    lines_out.append(f"  {class_b}")
    lines_out.append(f"Class (c) Only runtime_settings / global removal: {len(class_c)}")
    lines_out.append(f"  {class_c}")
    lines_out.append(f"Class (d) Anything else: {len(class_d)}")
    lines_out.append(f"  {[d[0] for d in class_d]}")

    lines_out.append("\n" + "=" * 80)
    lines_out.append("CLASS (D) UNIFIED DIFFS")
    lines_out.append("=" * 80)
    for name, lp, ll, np, nl, diff in class_d:
        lines_out.append(f"\n--- {name} ({lp}:{ll} vs {np}:{nl}) ---")
        lines_out.append(diff)

    lines_out.append("\n" + "=" * 80)
    lines_out.append("PROCESS_MESSAGE-RELATED FILES (services/chat/*) UNIFIED DIFFS")
    lines_out.append("=" * 80)
    for name, lp, ll, np, nl, diff in chat_related_diffs:
        lines_out.append(f"\n--- {name} ({lp}:{ll} vs {np}:{nl}) ---")
        lines_out.append(diff)

    os.makedirs("scratch/logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"scratch/logs/ast_compare_{ts}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))

    print(f"Saved to {out_path}")
    print("\n".join(lines_out[:25]))


if __name__ == "__main__":
    main()
