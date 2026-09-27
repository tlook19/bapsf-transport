"""Reference-route line coverage, and a check that a diff cannot reach it.

The golden route is the configuration ``baseline_sim1d.build_baseline_config``
builds, run at its full horizon. This tool records which lines of ``cablp/``
that run executes, then asks of a diff whether it changes only code the run
never executes. A diff that changes no executed line cannot move the reference
trajectory through the code it changes; configuration resolution, the
configuration identity and the untraced surfaces listed in the printed caveat
still need their own checks.

Two subcommands::

    # one map per kernel route (pure and compiled), from a clean tree
    python scripts/gates/reference_coverage.py capture --outdir <dir>

    # classify each hunk of a diff against the union of the maps
    python scripts/gates/reference_coverage.py check \
        --map <pure.json> --map <compiled.json> <base>..<head>

``capture`` runs each route in its own process with ``sys.monitoring``
(PEP 669) LINE events, which are enabled before ``cablp`` is imported. The
callback records the location and returns ``DISABLE``, so each line costs one
event and the run proceeds near native speed. Hits recorded while the route's
modules import are import-time lines; every later hit is a run-time line.
Cython (``.pyx``) code is not traced: the compiled route's map covers only the
Python it executes.

``check`` exits 0 with ``PROOF-ELIGIBLE`` when no hunk is REACHED, 1 with
``GOLDEN-REQUIRED`` otherwise, and 2 when it refuses (a map that does not
describe the base, or a missing route).
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = 1
ROUTES = ("pure", "compiled")
UNTRACED_NOTE = (
    "Cython sources (.pyx/.pxd) and C extensions are not traced; a line map "
    "covers only the Python the route executes."
)

# Paths outside cablp/*.py that can change the reference run without changing
# a traced line. A diff touching any of them needs the golden.
GOLDEN_PATH_RULES = (
    (re.compile(r".*\.(pyx|pxd|pxi|c|h|cpp)$"), "compiled source (not traced)"),
    (re.compile(r"^build_ext\.py$"), "extension build script"),
    (re.compile(r"^(pyproject\.toml|poetry\.lock|setup\.py|setup\.cfg|"
                r"MANIFEST\.in)$"), "packaging file"),
    (re.compile(r"^scripts/stances/"), "configuration file"),
    (re.compile(r"^scripts/baselines/"), "golden fixture"),
)
# Paths that cannot reach the run at all.
INERT_PATH = re.compile(
    r"(.*\.(md|rst)$|^tests/|^LICENSE|^\.gitignore$|^\.gitattributes$|"
    r"^\.pre-commit-config\.yaml$|^\.github/)"
)

CAVEAT = """\
CAVEAT. This check covers the Python lines of cablp/ on the reference route and
nothing else.
- It does not cover configuration resolution. Pair it with the no-solve
  config-diff preflight (scripts/gates/preflight_diffcfg.py); a change to a
  configuration template or default still rotates the configuration identity,
  and the identity rotation is owed whatever this check says.
- Cython code (.pyx/.pxd) and C extensions are not traced; any change to them
  is GOLDEN-REQUIRED.
- The maps cover one configuration (the golden route) on two kernel routes.
  Code the campaign reaches through other configurations is not covered.
Known holes (a change can reach the run without changing an executed line):
- String-keyed dispatch and presence probes: getattr(obj, name) with a computed
  name, getattr(obj, "x", default) and hasattr(obj, "x") change result when an
  unexecuted attribute is added or removed (for example the getattr-by-name
  loops in solver.py's restart capture and results/io.py's dataset writers,
  and the geometry presence probes in validation.py and kinetic_dvm.py). The
  check guards deleted names that appear literally on an executed line; it
  cannot see a name built at run time.
- Datasets written only when present: a writer that emits a dataset when an
  attribute exists or is not None changes the saved file, but not the
  trajectory, when unexecuted code that sets the attribute is added or removed.
- Dict and namespace iteration order: deleting or adding an unexecuted def,
  class attribute or registry entry changes the order of module or class
  __dict__, dir() and any registry, which code iterating them can observe.
- Package-level lazy loading (cablp/__init__.py __getattr__) can import a new
  module by attribute name; new modules are classified as unreached.
- Compile-time effects of unexecuted code are checked only for yield, await,
  global, nonlocal and __future__ tokens; other scope effects (a deleted
  unexecuted assignment changing a name from local to global) are not.
- Line numbers of executed code shift when unexecuted lines above them are
  deleted; anything that records line numbers (warnings, tracebacks, logged
  locations) changes.
- Classes used without executing any of their own lines (isinstance tests, an
  exception class raised or caught, generated dataclass methods) look
  import-only; the name guard catches only literal references on executed
  lines.
"""


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------

def _git(repo, *args, text=True):
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=text,
    )
    return out.stdout


def _git_show(repo, rev, path):
    """Return the bytes of ``path`` at ``rev``, or None when it is absent."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"],
        capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _tree_is_dirty(repo):
    """Return a list of reasons the tree is not the commit, empty when clean."""
    reasons = []
    tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked.strip():
        reasons.append("tracked changes:\n" + tracked.rstrip())
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard",
                     "--", "cablp", "scripts")
    code = [p for p in untracked.splitlines()
            if p.endswith((".py", ".pyx", ".pxd", ".toml"))]
    if code:
        reasons.append("untracked code files:\n" + "\n".join(code))
    return reasons


# --------------------------------------------------------------------------
# capture: the traced child process
# --------------------------------------------------------------------------

def _trace_child(args):
    """Run one route under sys.monitoring and write its raw map (child mode)."""
    repo = Path(args.repo).resolve()
    cablp_dir = str(repo / "cablp") + os.sep
    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "reference_coverage")
    import_hits = {}
    run_hits = {}
    phase = [import_hits]
    disable = mon.DISABLE

    def on_line(code, line):
        filename = code.co_filename
        if filename.startswith(cablp_dir):
            phase[0].setdefault(filename, set()).add(line)
        return disable

    mon.register_callback(tool, mon.events.LINE, on_line)

    opened = set()

    def on_audit(event, audit_args):
        if event == "open" and audit_args and isinstance(audit_args[0], str):
            opened.add(audit_args[0])

    sys.addaudithook(on_audit)
    mon.set_events(tool, mon.events.LINE)

    t0 = time.perf_counter()
    sys.path.insert(0, str(repo / "scripts" / "gates"))
    import baseline_sim1d as gate  # noqa: E402  (traced import)
    from cablp.cathode import kernels as selector  # noqa: E402
    import cablp  # noqa: E402
    if not os.path.realpath(cablp.__file__).startswith(cablp_dir):
        raise RuntimeError(f"cablp imported from {cablp.__file__}, not {repo}")
    t_import = time.perf_counter()
    phase[0] = run_hits
    params, flags = gate.build_baseline_config()
    _, _, summary, _ = gate.run_baseline(params, flags)
    t_run = time.perf_counter()
    mon.set_events(tool, 0)
    mon.free_tool_id(tool)

    import numpy
    identity = gate.baseline_lineage(params, flags).identity
    loaded = []
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            loaded.append(path)
    payload = {
        "route": args.route,
        "kernel_provenance": selector.PROVENANCE,
        "configuration_identity": identity,
        "python": sys.version,
        "numpy": numpy.__version__,
        "steps": int(summary.steps),
        "final_time_s": float(summary.final_time),
        "wall_s_import": t_import - t0,
        "wall_s_run": t_run - t_import,
        "import_hits": {k: sorted(v) for k, v in import_hits.items()},
        "run_hits": {k: sorted(v) for k, v in run_hits.items()},
        "loaded_files": sorted(set(loaded)),
        "opened_files": sorted(opened),
    }
    Path(args.out).write_text(json.dumps(payload))
    print(f"route {args.route}: kernels={selector.PROVENANCE} "
          f"steps={summary.steps} final_time={summary.final_time:.6e} s "
          f"run_wall={t_run - t_import:.1f} s", flush=True)
    return 0


def _repo_relative(repo, path):
    try:
        return Path(os.path.realpath(path)).relative_to(repo).as_posix()
    except ValueError:
        return None


def _build_map(repo, raw, commit, wall_total, command):
    """Turn a child's raw hits into the committed map schema."""
    tracked = set(_git(repo, "ls-files").splitlines())
    files = {}
    for rel in sorted(p for p in tracked
                      if p.startswith("cablp/") and p.endswith(".py")):
        files[rel] = {
            "sha256": _sha256((repo / rel).read_bytes()),
            "import_lines": [],
            "run_lines": [],
        }
    for key, field in (("import_hits", "import_lines"),
                       ("run_hits", "run_lines")):
        for path, lines in raw[key].items():
            rel = _repo_relative(repo, path)
            if rel is None or rel not in files:
                raise RuntimeError(f"traced file is not a tracked cablp file: {path}")
            files[rel][field] = lines
    route_files = {}
    for path in raw["loaded_files"] + raw["opened_files"]:
        rel = _repo_relative(repo, path)
        if rel is None or rel in files or rel.startswith(".git/") or \
                rel == _repo_relative(repo, __file__):
            continue
        if not (repo / rel).is_file():
            continue
        if rel.endswith(".py"):
            kind = "python-untraced"
        elif rel.endswith((".so", ".pyd")):
            kind = "extension"
        elif rel.endswith(".pyc") or "__pycache__" in rel:
            continue
        else:
            kind = "data"
        route_files[rel] = {
            "sha256": _sha256((repo / rel).read_bytes()),
            "kind": kind,
            "tracked": rel in tracked,
        }
    n_import = sum(len(v["import_lines"]) for v in files.values())
    n_run = sum(len(v["run_lines"]) for v in files.values())
    return {
        "tool": "reference_coverage",
        "schema": SCHEMA,
        "route": raw["route"],
        "kernel_provenance": raw["kernel_provenance"],
        "commit": commit,
        "dirty": False,
        "configuration_identity": raw["configuration_identity"],
        "python": raw["python"],
        "numpy": raw["numpy"],
        "untraced": UNTRACED_NOTE,
        "command": command,
        "steps": raw["steps"],
        "final_time_s": raw["final_time_s"],
        "wall_s_import": raw["wall_s_import"],
        "wall_s_run": raw["wall_s_run"],
        "wall_s_process": wall_total,
        "files_traced": sum(1 for v in files.values()
                            if v["import_lines"] or v["run_lines"]),
        "lines_import": n_import,
        "lines_run": n_run,
        "files": files,
        "route_files": route_files,
    }


def capture(args):
    repo = Path(args.repo).resolve()
    outdir = Path(args.outdir).resolve()
    if outdir == repo or repo in outdir.parents:
        print(f"refusing: --outdir {outdir} is inside the repository")
        return 2
    dirty = _tree_is_dirty(repo)
    if dirty:
        print("refusing to capture on a dirty tree:\n" + "\n".join(dirty))
        return 2
    commit = _git(repo, "rev-parse", "HEAD").strip()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"reference_coverage_{commit[:7]}"
    parent_cmd = (f"cd {shlex.quote(str(Path.cwd()))} && "
                  + " ".join(shlex.quote(a) for a in sys.argv))
    procs = {}
    for route in args.routes:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)
        env.pop("CABLP_COMPILED_KERNELS", None)
        if route == "compiled":
            env["CABLP_COMPILED_KERNELS"] = "1"
        raw_path = outdir / f"{stem}_{route}.raw.json"
        log_path = outdir / f"{stem}_{route}.log"
        child = [sys.executable, str(Path(__file__).resolve()), "_trace",
                 "--repo", str(repo), "--route", route, "--out", str(raw_path)]
        env_prefix = f"PYTHONPATH={shlex.quote(str(repo))} " + (
            "CABLP_COMPILED_KERNELS=1 " if route == "compiled"
            else "env -u CABLP_COMPILED_KERNELS ")
        child_cmd = env_prefix + " ".join(shlex.quote(a) for a in child)
        (outdir / f"{stem}_{route}.cmd").write_text(
            f"# capture command\n{parent_cmd}\n"
            f"# traced child for route {route}\n{child_cmd}\n")
        log = open(log_path, "w")
        procs[route] = (subprocess.Popen(child, env=env, cwd=str(repo),
                                         stdout=log, stderr=subprocess.STDOUT),
                        time.perf_counter(), raw_path, log, child_cmd)
        print(f"started route {route}: log {log_path}", flush=True)
    limit_s = args.abort_after_min * 60.0
    walls = {}
    rc = 0
    while procs:
        time.sleep(10)
        for route in list(procs):
            proc, t0, raw_path, log, child_cmd = procs[route]
            elapsed = time.perf_counter() - t0
            if proc.poll() is None:
                if elapsed > limit_s:
                    proc.kill()
                    proc.wait()
                    log.close()
                    print(f"route {route}: ABORTED at {elapsed / 60:.1f} min "
                          f"(limit {args.abort_after_min} min); no map written",
                          flush=True)
                    rc = 1
                    del procs[route]
                continue
            log.close()
            del procs[route]
            walls[route] = elapsed
            if proc.returncode != 0:
                print(f"route {route}: FAILED exit={proc.returncode} after "
                      f"{elapsed:.1f} s", flush=True)
                rc = 1
                continue
            raw = json.loads(raw_path.read_text())
            payload = _build_map(repo, raw, commit, elapsed,
                                 {"capture": parent_cmd, "child": child_cmd})
            if (route == "compiled") != ("cython" in payload["kernel_provenance"]):
                print(f"route {route}: wrong kernels loaded "
                      f"({payload['kernel_provenance']})", flush=True)
                rc = 1
                continue
            map_path = outdir / f"{stem}_{route}.json"
            map_path.write_text(json.dumps(payload, indent=1, sort_keys=True)
                                + "\n")
            raw_path.unlink()
            print(f"route {route}: wall {elapsed:.1f} s, "
                  f"{payload['files_traced']} files traced, "
                  f"{payload['lines_import']} import-time + "
                  f"{payload['lines_run']} run-time lines -> {map_path}",
                  flush=True)
    print(UNTRACED_NOTE)
    return rc


# --------------------------------------------------------------------------
# check: source analysis
# --------------------------------------------------------------------------

_COMPILE_EFFECT = {"yield", "await", "global", "nonlocal", "__future__"}
_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=]")


class Source:
    """One side of a file: text, AST, code lines, own lines per statement."""

    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines()
        self.tree = ast.parse(text) if text else ast.Module(body=[], type_ignores=[])
        self.code_lines = self._code_lines(text)
        self.event_lines = self._event_lines(text)
        self.parent = {}
        self.stmts = []
        self._index(self.tree, None)
        # innermost statement owning each line (own lines only)
        self.owner = {}
        for stmt in sorted(self.stmts, key=lambda s: _depth(s, self.parent)):
            for line in self.own_lines(stmt):
                self.owner[line] = stmt

    @staticmethod
    def _code_lines(text):
        out = set()
        skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
                tokenize.ENDMARKER}
        for tok in tokenize.tokenize(io.BytesIO(text.encode()).readline):
            if tok.type not in skip:
                out.update(range(tok.start[0], tok.end[0] + 1))
        for i, line in enumerate(text.splitlines()[:2], 1):
            if _COOKIE.match(line):
                out.add(i)
        return out

    @staticmethod
    def _event_lines(text):
        out = set()
        if not text:
            return out
        stack = [compile(text, "<old>", "exec", dont_inherit=True)]
        while stack:
            code = stack.pop()
            for _, _, line in code.co_lines():
                if line is not None:
                    out.add(line)
            stack.extend(c for c in code.co_consts if hasattr(c, "co_lines"))
        return out

    def _index(self, node, parent_stmt):
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, ast.stmt):
                    self.parent[item] = (node, field)
                    self.stmts.append(item)
                    self._index(item, item)
                elif isinstance(item, (ast.excepthandler, ast.match_case)):
                    self.parent[item] = (node, field)
                    self._index(item, parent_stmt)
                elif isinstance(item, ast.AST):
                    self._index(item, parent_stmt)

    def own_lines(self, stmt):
        """Lines of ``stmt`` not inside a child statement's span."""
        lines = set(range(stmt.lineno, stmt.end_lineno + 1))
        for dec in getattr(stmt, "decorator_list", []):
            lines.update(range(dec.lineno, dec.end_lineno + 1))
        for child in _child_stmts(stmt):
            lines.difference_update(range(child.lineno, child.end_lineno + 1))
        return lines

    def span(self, stmt):
        start = min([stmt.lineno] + [d.lineno for d in
                                     getattr(stmt, "decorator_list", [])])
        return set(range(start, stmt.end_lineno + 1))

    def enclosing_stmt(self, node):
        """The statement (or module) owning the block ``node`` sits in."""
        owner, _ = self.parent[node]
        while not isinstance(owner, (ast.stmt, ast.Module)):
            owner, _ = self.parent[owner]
        return owner

    def block_of(self, stmt):
        owner, field = self.parent[stmt]
        return owner, getattr(owner, field)

    def body_lines(self, fn):
        lines = set()
        for child in fn.body:
            lines |= set(range(child.lineno, child.end_lineno + 1))
        return lines

    def qualname(self, stmt):
        parts = [stmt.name]
        node = stmt
        while node in self.parent:
            owner = self.enclosing_stmt(node)
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                parts.append(owner.name)
            node = owner
            if isinstance(owner, ast.Module):
                break
        return ".".join(reversed(parts))

    def defs(self):
        return {self.qualname(s): s for s in self.stmts
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef))}

    def docstring_owner(self, stmt):
        """Return the def/class/module whose docstring ``stmt`` is, else None."""
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            return None
        owner, field = self.parent[stmt]
        if field == "body" and getattr(owner, "body", [None])[0] is stmt and \
                isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                   ast.AsyncFunctionDef)):
            return owner
        return None

    def scope_bindings(self, block_owner):
        """Names bound in the scope whose body is ``block_owner.body``."""
        counts = {}

        def visit(stmts):
            for s in stmts:
                names = []
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                    names = [s.name]
                elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                    for t in targets:
                        names += [n.id for n in ast.walk(t)
                                  if isinstance(n, ast.Name)]
                elif isinstance(s, (ast.Import, ast.ImportFrom)):
                    names = [(a.asname or a.name).split(".")[0] for a in s.names]
                for n in names:
                    counts[n] = counts.get(n, 0) + 1
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    for child in _child_blocks(s):
                        visit(child)

        visit(block_owner.body)
        return counts


def _child_blocks(stmt):
    for field in ("body", "orelse", "finalbody"):
        block = getattr(stmt, field, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
    for handler in getattr(stmt, "handlers", []):
        yield handler.body
    for case in getattr(stmt, "cases", []):
        yield case.body


def _child_stmts(stmt):
    for block in _child_blocks(stmt):
        yield from block


def _depth(stmt, parent):
    depth = 0
    node = stmt
    while node in parent:
        node = parent[node][0]
        depth += 1
    return depth


def _is_def(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _has_call(node):
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def _tokens_on_lines(text):
    """Map line -> set of NAME tokens and exact string-literal values."""
    out = {}
    for tok in tokenize.tokenize(io.BytesIO(text.encode()).readline):
        if tok.type == tokenize.NAME:
            value = tok.string
        elif tok.type == tokenize.STRING:
            try:
                value = ast.literal_eval(tok.string)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(value, str):
                continue
        else:
            continue
        for line in range(tok.start[0], tok.end[0] + 1):
            out.setdefault(line, set()).add(value)
    return out


class Coverage:
    """Union of the maps' executed lines, per file."""

    def __init__(self, maps):
        self.run = {}
        self.imp = {}
        for m in maps:
            for rel, entry in m["files"].items():
                self.run.setdefault(rel, set()).update(entry["run_lines"])
                self.imp.setdefault(rel, set()).update(entry["import_lines"])
        for rel in self.imp:
            self.imp[rel] -= self.run.get(rel, set())

    def phase(self, rel, lines):
        """'run', 'import' or None for the most-executed of ``lines``."""
        run = self.run.get(rel, set())
        imp = self.imp.get(rel, set())
        if any(line in run for line in lines):
            return "run"
        if any(line in imp for line in lines):
            return "import"
        return None


class NameGuard:
    """Names and strings appearing on executed lines of the base tree."""

    def __init__(self, repo, base, coverage, route_python):
        self.hits = {}
        for rel in set(coverage.run) | set(coverage.imp):
            executed = coverage.run.get(rel, set()) | coverage.imp.get(rel, set())
            if not executed:
                continue
            data = _git_show(repo, base, rel)
            if data is None:
                continue
            for line, names in _tokens_on_lines(data.decode()).items():
                if line in executed:
                    for name in names:
                        self.hits.setdefault(name, []).append((rel, line))
        for rel in route_python:
            data = _git_show(repo, base, rel)
            if data is None:
                continue
            for line, names in _tokens_on_lines(data.decode()).items():
                for name in names:
                    self.hits.setdefault(name, []).append((rel, line))

    def references(self, name, rel, exclude):
        return [(r, n) for r, n in self.hits.get(name, [])
                if not (r == rel and n in exclude)]


class FileCheck:
    """Classify the hunks of one cablp .py file."""

    def __init__(self, rel, old_text, new_text, hunks, cov, guard):
        self.rel = rel
        self.old = Source(old_text)
        self.new = Source(new_text)
        self.hunks = hunks
        self.cov = cov
        self.guard = guard
        self.deleted = set()
        self.added = set()
        for a, b, c, d in hunks:
            self.deleted.update(range(a, a + b))
            self.added.update(range(c, c + d))
        self.old_defs = self.old.defs()
        self.new_defs = self.new.defs()
        self.old_def_q = {id(s): q for q, s in self.old_defs.items()}

    # -- old side ----------------------------------------------------------

    def classify_old_line(self, line):
        """Return (verdict, reason) for a changed or deleted old line."""
        if line not in self.old.code_lines:
            return "UNREACHED", None
        stmt = self.old.owner.get(line)
        if stmt is None:
            return "REACHED", f"old line {line} has no owning statement"
        doc_owner = self.old.docstring_owner(stmt)
        if doc_owner is not None:
            return self._docstring_verdict(doc_owner, line)
        phase = self.cov.phase(self.rel, self.old.own_lines(stmt))
        if phase == "run":
            return "REACHED", f"old line {line} executed at run time"
        if phase is None:
            return self._compile_effect(stmt, line)
        return self._import_role(stmt, line)

    def _docstring_verdict(self, owner, line):
        if isinstance(owner, ast.Module):
            return "REACHED", f"old line {line}: module docstring (binds __doc__)"
        if self.cov.phase(self.rel, self.old.span(owner)) == "run" or (
                _is_def(owner) and self.cov.phase(
                    self.rel, self.old.body_lines(owner))):
            return "REACHED", (f"old line {line}: docstring of executed "
                               f"{owner.name}")
        if isinstance(owner, ast.ClassDef):
            return "IMPORT-ONLY", f"old line {line}: docstring of {owner.name}"
        return "UNREACHED", None

    def _compile_effect(self, stmt, line):
        hit = set(re.findall(r"\w+", self.old.lines[line - 1])) & _COMPILE_EFFECT
        if not hit:
            return "UNREACHED", None
        if "__future__" in hit:
            return "REACHED", f"old line {line}: __future__ import"
        scope = stmt
        while not isinstance(scope, ast.Module) and not _is_def(scope):
            scope = self.old.enclosing_stmt(scope)
        if _is_def(scope) and self.old.span(scope) <= self.deleted:
            return "UNREACHED", None
        return "REACHED", (f"old line {line}: unexecuted {sorted(hit)} changes "
                           "how its enclosing scope compiles")

    def _import_role(self, stmt, line):
        """Classify an import-time-only line by what it binds."""
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in stmt.decorator_list:
                if dec.lineno <= line <= dec.end_lineno:
                    if _has_call(dec):
                        return "REACHED", (f"old line {line}: decorator call on "
                                           f"{stmt.name} (registry entry)")
                    break
            if _is_def(stmt):
                args = stmt.args
                for default in args.defaults + [d for d in args.kw_defaults if d]:
                    if default.lineno <= line <= default.end_lineno:
                        return "REACHED", (f"old line {line}: default argument "
                                           f"of {stmt.name}")
            return self._header_verdict(stmt, f"old line {line}")
        return "REACHED", (f"old line {line}: import-time "
                           f"{type(stmt).__name__} (defines data)")

    def _header_verdict(self, stmt, label):
        """An old def/class header is import-only when nothing inside it ran."""
        src = self.old
        inside = src.span(stmt) - src.own_lines(stmt)
        if self.cov.phase(self.rel, inside) == "run":
            return "REACHED", f"{label}: header of {stmt.name}, whose body runs"
        if _is_def(stmt) and self.cov.phase(self.rel, src.body_lines(stmt)):
            return "REACHED", (f"{label}: header of {stmt.name}, whose body "
                               "executes")
        if isinstance(stmt, ast.ClassDef):
            for child in ast.walk(stmt):
                if isinstance(child, ast.stmt) and child is not stmt and \
                        not _is_def(child) and \
                        src.docstring_owner(child) is None and \
                        self.cov.phase(self.rel, src.own_lines(child)):
                    return "REACHED", (f"{label}: class {stmt.name} defines "
                                       "executed data")
        q = self.old_def_q.get(id(stmt))
        if q is not None and q not in self.new_defs:
            refs = self.guard.references(stmt.name, self.rel, src.span(stmt))
            if refs:
                r, n = refs[0]
                return "REACHED", (f"{label}: removes or renames {q}, whose "
                                   f"name appears on executed line {r}:{n}")
        return "IMPORT-ONLY", f"{label}: header of unexecuted {stmt.name}"

    # -- new side ----------------------------------------------------------

    def map_new_to_old(self, line):
        """Old line number of an unchanged new line, else None."""
        if line in self.added:
            return None
        shift = 0
        for a, b, c, d in self.hunks:
            before = (c + d - 1 < line) if d else (c < line)
            if before:
                shift += b - d
        return line + shift

    def _old_stmt_for_new(self, stmt):
        """The old statement an unchanged new statement corresponds to."""
        for line in sorted(self.new.own_lines(stmt)):
            if line in self.new.code_lines:
                old = self.map_new_to_old(line)
                if old is not None:
                    return self.old.owner.get(old)
        return None

    def block_reach(self, block):
        """'run', 'import', 'none' (never entered) or None (unknown)."""
        first_known = None
        result = None
        for stmt in block:
            old = self._old_stmt_for_new(stmt)
            if old is None:
                continue
            own = self.old.own_lines(old)
            phase = self.cov.phase(self.rel, own)
            if phase == "run":
                return "run"
            if phase == "import":
                result = "import"
            if first_known is None:
                first_known = bool(own & self.old.event_lines)
        if result:
            return result
        if first_known:
            return "none"
        return None

    def classify_new_line(self, line):
        if line not in self.new.code_lines:
            return "UNREACHED", None
        stmt = self.new.owner.get(line)
        if stmt is None:
            return "REACHED", f"new line {line} has no owning statement"
        node = stmt
        while True:
            owner, block = self.new.block_of(node)
            if isinstance(owner, (ast.excepthandler, ast.match_case)):
                owner = self.new.enclosing_stmt(owner)
            reach = self.block_reach(block)
            if reach == "run":
                return "REACHED", f"new line {line}: its block runs"
            if reach == "none":
                return "UNREACHED", None
            if reach == "import":
                return self._new_import_role(node, stmt, line)
            # unknown: decide from the owner of the block
            if isinstance(owner, ast.Module):
                if self.old.text:
                    return self._new_import_role(node, stmt, line)
                return "UNREACHED", None
            old_owner = self._old_stmt_for_new(owner)
            if _is_def(owner):
                if old_owner is None:
                    if self._new_def_ok(owner, line):
                        return "UNREACHED", None
                    return "REACHED", f"new line {line}: body of new {owner.name}"
                phase = self.cov.phase(self.rel, self.old.body_lines(old_owner)) \
                    if _is_def(old_owner) else "run"
                if phase:
                    return "REACHED", (f"new line {line}: body of executed "
                                       f"{owner.name}")
                return "UNREACHED", None
            if old_owner is not None:
                phase = self.cov.phase(self.rel, self.old.own_lines(old_owner))
                if phase is None and bool(self.old.own_lines(old_owner)
                                          & self.old.event_lines):
                    return "UNREACHED", None
                return "REACHED", (f"new line {line}: inside a block whose "
                                   "entry cannot be placed")
            node = owner

    def _new_def_ok(self, fn, line):
        """A wholly new def binds a fresh name and evaluates nothing at import."""
        if any(_has_call(d) for d in fn.decorator_list):
            return False
        if _is_def(fn) and any(_has_call(d) for d in
                               fn.args.defaults + [k for k in fn.args.kw_defaults if k]):
            return False
        owner = self.new.enclosing_stmt(fn)
        if not isinstance(owner, (ast.Module, ast.ClassDef)):
            return True
        if self.new.scope_bindings(owner).get(fn.name, 0) > 1:
            return False
        return not self.guard.references(fn.name, None, set())

    def _new_import_role(self, top, stmt, line):
        """A new line at module or class level runs at import."""
        doc_owner = self.new.docstring_owner(stmt)
        if doc_owner is not None and not isinstance(doc_owner, ast.Module):
            old_owner = self._old_stmt_for_new(doc_owner)
            if old_owner is None:
                return "IMPORT-ONLY", f"new line {line}: docstring of new {doc_owner.name}"
            return self._docstring_verdict(old_owner, line)
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            header = top.body[0].lineno > line or any(
                d.lineno <= line <= d.end_lineno for d in top.decorator_list)
            old_def = self._old_stmt_for_new(top)
            if old_def is None:
                if isinstance(top, ast.ClassDef):
                    return "REACHED", f"new line {line}: new class {top.name} runs its body at import"
                if self._new_def_ok(top, line):
                    return "IMPORT-ONLY", f"new line {line}: new def {top.name}"
                return "REACHED", (f"new line {line}: new def {top.name} "
                                   "rebinds a name, calls at import, or is "
                                   "named on an executed line")
            if header:
                for dec in top.decorator_list:
                    if dec.lineno <= line <= dec.end_lineno and _has_call(dec):
                        return "REACHED", f"new line {line}: decorator call on {top.name}"
                if _is_def(top):
                    for default in top.args.defaults + [
                            k for k in top.args.kw_defaults if k]:
                        if default.lineno <= line <= default.end_lineno:
                            return "REACHED", f"new line {line}: default argument of {top.name}"
                if self.new.scope_bindings(self.new.enclosing_stmt(top)).get(
                        top.name, 0) > 1:
                    return "REACHED", f"new line {line}: {top.name} is bound twice"
                return self._header_verdict(old_def, f"new line {line}")
        return "REACHED", (f"new line {line}: import-time "
                           f"{type(top).__name__} (defines data)")

    # -- hunks -------------------------------------------------------------

    def classify(self):
        results = []
        for a, b, c, d in self.hunks:
            verdicts = []
            for line in range(a, a + b):
                verdicts.append(self.classify_old_line(line))
            for line in range(c, c + d):
                verdicts.append(self.classify_new_line(line))
            reached = [r for v, r in verdicts if v == "REACHED"]
            imports = [r for v, r in verdicts if v == "IMPORT-ONLY"]
            if reached:
                verdict, reasons = "REACHED", reached
            elif imports:
                verdict, reasons = "IMPORT-ONLY", imports
            else:
                verdict, reasons = "UNREACHED", []
            results.append((verdict, a, b, c, d, reasons))
        return results


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _hunks(repo, base, head, path):
    diff = _git(repo, "diff", "-U0", "--no-renames", "--no-ext-diff",
                base, head, "--", path)
    hunks = []
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            a, b, c, d = m.groups()
            b = 1 if b is None else int(b)
            d = 1 if d is None else int(d)
            hunks.append((int(a), b, int(c), d))
    return hunks


def check(args):
    repo = Path(args.repo).resolve()
    if ".." not in args.range:
        print(f"refusing: expected <base>..<head>, got {args.range!r}")
        return 2
    base_ref, head_ref = args.range.split("..", 1)
    base = _git(repo, "rev-parse", "--verify", base_ref + "^{commit}").strip()
    head = _git(repo, "rev-parse", "--verify", head_ref + "^{commit}").strip()
    maps = [json.loads(Path(p).read_text()) for p in args.map]
    routes = sorted(m["route"] for m in maps)
    if not set(ROUTES) <= set(routes):
        print(f"refusing: maps cover routes {routes}; both {list(ROUTES)} "
              "are required")
        return 2
    identities = {m["configuration_identity"] for m in maps}
    if len(identities) != 1:
        print(f"refusing: maps disagree on configuration identity {identities}")
        return 2

    changes = _git(repo, "diff", "--no-renames", "--name-status", base, head)
    touched = []
    for row in changes.splitlines():
        status, path = row.split("\t", 1)
        touched.append((status[0], path))

    for m in maps:
        if m["commit"] == base:
            continue
        recorded = {**m["files"], **m["route_files"]}
        bad = []
        for status, path in touched:
            data = _git_show(repo, base, path)
            if data is None:
                continue
            if path in recorded:
                if recorded[path]["sha256"] != _sha256(data):
                    bad.append(f"{path} (sha256 differs)")
            elif path.startswith("cablp/") and path.endswith(".py"):
                bad.append(f"{path} (not in the map)")
        if bad:
            print(f"refusing: map {m['route']} is at {m['commit'][:12]}, not "
                  f"the base {base[:12]}, and these touched files differ "
                  "from it:\n  " + "\n  ".join(bad))
            return 2

    cov = Coverage(maps)
    route_files = {}
    for m in maps:
        route_files.update(m["route_files"])
    route_python = [p for p, e in route_files.items()
                    if e["kind"] == "python-untraced" and e.get("tracked", True)]
    guard = NameGuard(repo, base, cov, route_python)

    out_of_scope = []
    off_route = []
    inert = []
    rows = []
    for status, path in touched:
        if path.startswith("cablp/") and path.endswith(".py"):
            old = _git_show(repo, base, path)
            new = _git_show(repo, head, path)
            old_text = old.decode() if old is not None else ""
            new_text = new.decode() if new is not None else ""
            try:
                fc = FileCheck(path, old_text, new_text,
                               _hunks(repo, base, head, path), cov, guard)
                rows.extend((path,) + r for r in fc.classify())
            except SyntaxError as error:
                out_of_scope.append((path, f"does not parse ({error.msg})"))
            continue
        if path in route_files:
            out_of_scope.append(
                (path, f"on the reference route, not line-traced "
                       f"({route_files[path]['kind']})"))
            continue
        rule = next((why for pat, why in GOLDEN_PATH_RULES if pat.match(path)),
                    None)
        if rule:
            out_of_scope.append((path, rule))
        elif path.startswith("cablp/") and not INERT_PATH.match(path):
            out_of_scope.append((path, "non-Python file inside the package"))
        elif path.startswith("scripts/") and path.endswith(".py"):
            off_route.append(path)
        elif INERT_PATH.match(path) or path.startswith("scripts/") and \
                path.endswith(".md"):
            inert.append(path)
        else:
            out_of_scope.append((path, "unclassified path"))

    print(f"reference_coverage check {base[:12]}..{head[:12]}")
    print(f"maps: " + ", ".join(f"{m['route']}@{m['commit'][:12]}" for m in maps)
          + f"; configuration identity {identities.pop()[:12]}")

    def _span(a, b, c, d):
        old = f"-{a},{b}" if b else f"-{a},0"
        new = f"+{c},{d}" if d else f"+{c},0"
        return f"{old} {new}"

    counts = {"UNREACHED": 0, "IMPORT-ONLY": 0, "REACHED": 0}
    for row in rows:
        counts[row[1]] += 1
    print(f"hunks under cablp/: {len(rows)} "
          + " ".join(f"{k}={v}" for k, v in counts.items()))
    if off_route:
        print("off-route scripts (no effect on the reference run): "
              + ", ".join(off_route))
    if inert:
        print("inert paths: " + ", ".join(inert))
    if args.verbose:
        for path, verdict, a, b, c, d, _ in rows:
            if verdict == "UNREACHED":
                print(f"  UNREACHED   {path}:{a} ({_span(a, b, c, d)})")
    imports = [r for r in rows if r[1] == "IMPORT-ONLY"]
    if imports:
        print("IMPORT-ONLY hunks:")
        for path, _, a, b, c, d, reasons in imports:
            print(f"  {path}:{a} ({_span(a, b, c, d)}): {reasons[0]}")
    reached = [r for r in rows if r[1] == "REACHED"]
    verdict = "PROOF-ELIGIBLE" if not reached and not out_of_scope \
        else "GOLDEN-REQUIRED"
    print(f"VERDICT: {verdict}")
    for path, why in out_of_scope:
        print(f"  OUT-OF-SCOPE {path}: {why}")
    for path, _, a, b, c, d, reasons in reached:
        print(f"  REACHED {path}:{a} ({_span(a, b, c, d)}): {reasons[0]}"
              + (f" [+{len(reasons) - 1} more]" if len(reasons) > 1 else ""))
    print(CAVEAT, end="")
    return 0 if verdict == "PROOF-ELIGIBLE" else 1


# --------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Reference-route coverage maps and a diff-reachability check.")
    parser.add_argument("--repo", default=str(REPO),
                        help="repository root (default: this checkout)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="trace the golden route, both kernels")
    cap.add_argument("--outdir", required=True,
                     help="directory outside the repository for the maps")
    cap.add_argument("--routes", nargs="+", default=list(ROUTES), choices=ROUTES)
    cap.add_argument("--abort-after-min", type=float, default=50.0,
                     help="kill a route that runs longer than this (minutes)")
    chk = sub.add_parser("check", help="classify a diff against the maps")
    chk.add_argument("--map", action="append", required=True)
    chk.add_argument("range", help="<base>..<head>")
    chk.add_argument("--verbose", action="store_true",
                     help="also list UNREACHED hunks")
    trace = sub.add_parser("_trace", help=argparse.SUPPRESS)
    trace.add_argument("--route", required=True, choices=ROUTES)
    trace.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.cmd == "_trace":
        return _trace_child(args)
    if args.cmd == "capture":
        return capture(args)
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
