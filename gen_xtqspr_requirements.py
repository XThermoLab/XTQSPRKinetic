"""
Generate requirements files for XTQSPRKinetic-style projects.

This tool is designed for XTQSPRKinetic v2.7.0 and later, where the project can
be monolithic or split into src/main modules. It scans Python files recursively,
maps import names to distribution names, separates core and optional imports
where possible, and adds feature-specific dependencies that are not always
visible from static imports, such as openpyxl for Excel export and pyarrow for
Parquet descriptor caches.

Standard outputs:
  requirements.txt                         # core requirements
  requirements-optional.txt                # optional / feature-specific requirements
  requirements-all.txt                     # union of core + optional
  requirements-detected-report.json
  requirements-detected-report.md

Additional profile outputs:
  --profile py313-mordredcommunity
      requirements-all-py313-mordredcommunity.txt
      INSTALL_PY313_MORDREDCOMMUNITY.md

  --profile py313-original-mordred
      requirements-bootstrap-py313.txt
      requirements-all-update-py313-original-mordred.txt
      requirements-runtime-py313-original-mordred.txt
      INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md

  --profile py313-both
      writes both of the above profile groups

  --profile py313-mordred
      backward-compatible alias for py313-both

Examples:
  python gen_xtqspr_requirements.py ./src ./main --out-dir ./requirements_out
  python gen_xtqspr_requirements.py ./src ./main --xtqspr-defaults --pin none --profile py313-both

Pinning modes for standard outputs:
  --pin none        package names only
  --pin installed   pin only installed packages as name==version; keep uninstalled names unpinned
  --pin strict      require every detected package to be installed; fail if any version cannot be resolved

Mordred profiles:
  1) py313-mordredcommunity is the recommended clean Python 3.13 profile. It
     uses the community-maintained Mordred distribution while preserving the
     existing import path used by XTQSPRKinetic:
         from mordred import Calculator, descriptors
     It keeps NumPy 2.x compatibility for drfp 0.3.7.

  2) py313-original-mordred is a legacy compatibility profile for the original
     mordred==1.2.0 package. It reproduces the staged install procedure where
     broad Python 3.13 wheels are installed first, then numpy==1.26.4 and
     mordred==1.2.0 are forced. This profile is expected to be runtime-tested;
     pip check may report a drfp/numpy metadata conflict when drfp requires
     numpy>=2.2.2.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as md
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# -----------------------------------------------------------------------------
# Project-local modules and package-name mapping
# -----------------------------------------------------------------------------

LOCAL_IMPORT_TOPLEVELS = {
    "XTQSPRKinetic", "config", "utils", "pipeline", "evaluation", "featurization",
    "selection", "prediction", "shared_imports", "xtqspr_models", "autoselection",
    "auto_chooser_main", "main", "reaction_property_performance",
}

MANUAL_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "skfuzzy": "scikit-fuzzy",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "rdkit": "rdkit",
    "hdbscan": "hdbscan",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "shap": "shap",
    "boruta": "Boruta",
    "drfp": "drfp",
    # Default distribution name for the standard profile. The Python 3.13
    # mordredcommunity profile rewrites this distribution to mordredcommunity.
    "mordred": "mordred",
    "gensim": "gensim",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "openpyxl": "openpyxl",
    "lz4": "lz4",
    "zstandard": "zstandard",
    "pyarrow": "pyarrow",
    "setuptools": "setuptools",
    "Cython": "Cython",
    "cython": "Cython",
    "importlib_metadata": "importlib-metadata",
}

KNOWN_OPTIONAL_DISTS = {
    "lightgbm",           # alternative Boruta estimator
    "scikit-fuzzy",      # FCM topic/clustering paths
    "hdbscan",           # HDBSCAN topic/clustering paths
    "gensim",            # LDA coherence diagnostics
    "scipy",             # Hungarian alignment and scientific utilities
    "openpyxl",          # Excel output for reaction_property_performance.py
    "lz4",               # optional pipeline.pkl compression backend
    "zstandard",         # optional compression backend
    "importlib-metadata",# only needed on older Python versions
}

XTQSPR_OPTIONAL_EXTRAS = {
    "openpyxl",
    "lz4",
    "zstandard",
    "pyarrow",           # Parquet descriptor/cache support through pandas
}

XTQSPR_CORE_DEFAULTS = {
    "numpy", "pandas", "scikit-learn", "xgboost", "joblib", "PyYAML",
    "matplotlib", "shap", "Boruta", "drfp", "mordred", "rdkit",
}

# Build/bootstrap tools used by the Python 3.13 staged original-Mordred profile.
PY313_BOOTSTRAP_REQS = [
    "pip",
    "setuptools",
    "wheel",
    "Cython",
]

# Legacy original-Mordred runtime pair.
PY313_ORIGINAL_MORDRED_RUNTIME_REQS = [
    "mordred==1.2.0",
    "numpy==1.26.4",
]

# Broad first-stage extras for Python 3.13 legacy profile.
PY313_BROAD_EXTRA_DISTS = {
    "setuptools",
    "pyarrow",
}

# Install original Mordred only in the final runtime stage for deterministic behavior.
PY313_ORIGINAL_BROAD_EXCLUDE_DISTS = {
    "mordred",
}

# Recommended clean Python 3.13 community-Mordred overrides.
# These are written exactly as requirement lines in the special profile file.
PY313_MORDREDCOMMUNITY_OVERRIDES = {
    "mordred": "mordredcommunity",
    "mordredcommunity": "mordredcommunity",
    "numpy": "numpy>=2.2.2",
    "drfp": "drfp==0.3.7",
}

# Add these in clean community profile even if not detected statically.
PY313_MORDREDCOMMUNITY_EXTRA_DISTS = {
    "setuptools",
    "pyarrow",
}

EXTRA_STDLIB = {
    "__future__", "argparse", "ast", "collections", "contextlib", "copy", "dataclasses",
    "datetime", "errno", "functools", "hashlib", "importlib", "inspect", "itertools", "json",
    "logging", "math", "os", "pathlib", "pickle", "platform", "random", "re", "shutil",
    "statistics", "subprocess", "sys", "tempfile", "time", "traceback", "types",
    "typing", "typing_extensions", "warnings", "zipfile",
}


def _stdlib_modules() -> Set[str]:
    mods = set(getattr(sys, "stdlib_module_names", set()) or set())
    mods.update(EXTRA_STDLIB)
    return mods


@dataclass
class ImportRecord:
    module: str
    distribution: Optional[str]
    file: str
    line: int
    optional_context: bool
    reason: str


class ImportScanner(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.records: List[Tuple[str, int, bool]] = []
        self._try_depth = 0

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        self._try_depth += 1
        for child in node.body:
            self.visit(child)
        self._try_depth -= 1
        for group in (node.handlers, node.orelse, node.finalbody):
            for child in group:
                self.visit(child)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            top = alias.name.split(".", 1)[0]
            self.records.append((top, int(getattr(node, "lineno", 0)), self._try_depth > 0))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if int(getattr(node, "level", 0) or 0) > 0:
            return
        if node.module:
            top = node.module.split(".", 1)[0]
            self.records.append((top, int(getattr(node, "lineno", 0)), self._try_depth > 0))


def _iter_python_files(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    ignore_dirs = {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "venv", ".venv",
        "env", ".tox", "build", "dist", "site-packages",
    }
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            out.append(p)
        elif p.is_dir():
            for f in p.rglob("*.py"):
                if any(part in ignore_dirs for part in f.parts):
                    continue
                out.append(f)
    return sorted(set(out))


def _package_map() -> Dict[str, List[str]]:
    try:
        mp = md.packages_distributions()
        return {k: list(v) for k, v in mp.items()}
    except Exception:
        return {}


def _dist_for_import(mod: str, pkg_map: Dict[str, List[str]]) -> Optional[str]:
    if mod in MANUAL_IMPORT_TO_DIST:
        return MANUAL_IMPORT_TO_DIST[mod]
    vals = pkg_map.get(mod)
    if vals:
        return sorted(vals, key=lambda x: (len(x), x.lower()))[0]
    try:
        md.version(mod)
        return mod
    except Exception:
        return None


def _format_requirement(dist: str, pin: str) -> str:
    if pin == "none":
        return dist
    try:
        version = md.version(dist)
        return f"{dist}=={version}"
    except Exception:
        if pin == "strict":
            raise RuntimeError(f"Cannot resolve installed version for distribution: {dist}")
        return dist


def _format_report_distribution(dist: str) -> str:
    """Return a Markdown-friendly distribution label with the installed version when available."""
    try:
        version = md.version(dist)
        return f"{dist} `{version}`"
    except Exception:
        return f"{dist} `version unavailable`"


def scan(paths: Sequence[Path], xtqspr_defaults: bool = False) -> Dict[str, object]:
    stdlib = _stdlib_modules()
    pkg_map = _package_map()
    files = _iter_python_files(paths)
    records: List[ImportRecord] = []
    unresolved: Set[str] = set()
    local: Set[str] = set()
    std: Set[str] = set()

    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except UnicodeDecodeError:
            tree = ast.parse(f.read_text(errors="ignore"), filename=str(f))
        scanner = ImportScanner(f)
        scanner.visit(tree)
        for mod, line, opt in scanner.records:
            if mod in stdlib:
                std.add(mod)
                continue
            if mod in LOCAL_IMPORT_TOPLEVELS:
                local.add(mod)
                continue
            dist = _dist_for_import(mod, pkg_map)
            if not dist:
                unresolved.add(mod)
                continue
            reason = "optional_try_import" if opt else "direct_import"
            if dist in KNOWN_OPTIONAL_DISTS:
                reason = "known_xtqspr_optional"
                opt = True
            records.append(ImportRecord(mod, dist, str(f), line, opt, reason))

    core_dists = {r.distribution for r in records if r.distribution and not r.optional_context}
    opt_dists = {r.distribution for r in records if r.distribution and r.optional_context}
    opt_dists.update(XTQSPR_OPTIONAL_EXTRAS)

    if xtqspr_defaults:
        core_dists.update(XTQSPR_CORE_DEFAULTS)
        opt_dists.update(KNOWN_OPTIONAL_DISTS)
        opt_dists.update(XTQSPR_OPTIONAL_EXTRAS)

    opt_dists = set(opt_dists) - set(core_dists)

    return {
        "files_scanned": [str(f) for f in files],
        "core_distributions": sorted(core_dists, key=str.lower),
        "optional_distributions": sorted(opt_dists, key=str.lower),
        "all_distributions": sorted(set(core_dists) | set(opt_dists), key=str.lower),
        "records": [asdict(r) for r in records],
        "unresolved_import_modules": sorted(unresolved),
        "ignored_stdlib_modules": sorted(std),
        "ignored_local_modules": sorted(local),
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def _write_plain_req(path: Path, lines: Iterable[str]) -> None:
    cleaned = [str(x).strip() for x in lines if str(x).strip()]
    path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")


def _dedupe_sort_req_lines(lines: Iterable[str]) -> List[str]:
    """Sort requirement lines by package name while preserving specifiers."""
    cleaned = [str(x).strip() for x in lines if str(x).strip()]
    # If the same base package appears twice, keep the later occurrence so profile
    # overrides can replace generic package names.
    by_base: Dict[str, str] = {}
    for line in cleaned:
        base = line.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip().lower()
        by_base[base] = line
    return sorted(by_base.values(), key=lambda x: x.lower())


def write_standard_outputs(result: Dict[str, object], out_dir: Path, pin: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    core = [str(x) for x in result["core_distributions"]]
    opt = [str(x) for x in result["optional_distributions"]]
    all_d = [str(x) for x in result["all_distributions"]]

    def write_req(name: str, dists: Iterable[str]) -> None:
        lines = [_format_requirement(d, pin) for d in sorted(set(dists), key=str.lower)]
        _write_plain_req(out_dir / name, lines)

    write_req("requirements.txt", core)
    write_req("requirements-optional.txt", opt)
    write_req("requirements-all.txt", all_d)
    (out_dir / "requirements-detected-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_lines = [
        "# XTQSPRKinetic requirements detection report",
        "",
        f"Files scanned: {len(result['files_scanned'])}",
        f"Core distributions: {len(core)}",
        f"Optional distributions: {len(opt)}",
        "",
        "## Core",
        "",
        *[f"- {_format_report_distribution(d)}" for d in core],
        "",
        "## Optional / feature-specific",
        "",
        *[f"- {_format_report_distribution(d)}" for d in opt],
        "",
        "## Unresolved import modules",
        "",
        *[f"- {m}" for m in result["unresolved_import_modules"]],
        "",
        "## Notes",
        "",
        "- Optional dependencies are needed only for specific configurations/features.",
        "- RDKit installation name may depend on the package manager; conda-forge `rdkit` is often reliable.",
        "- Standard files reflect static import detection and the selected `--pin` mode.",
        "- For Python 3.13, use `--profile py313-mordredcommunity` for the clean community-Mordred profile.",
        "- Use `--profile py313-original-mordred` only for legacy original Mordred compatibility testing.",
    ]
    (out_dir / "requirements-detected-report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _community_profile_lines(result: Dict[str, object]) -> List[str]:
    """Build the clean Python 3.13 mordredcommunity requirements list."""
    dists = {str(x) for x in result["all_distributions"]}
    dists.update(PY313_MORDREDCOMMUNITY_EXTRA_DISTS)
    dists.add("mordredcommunity")

    lines: List[str] = []
    for dist in sorted(dists, key=str.lower):
        key = dist.lower()
        if key == "mordred":
            lines.append(PY313_MORDREDCOMMUNITY_OVERRIDES["mordred"])
        elif key == "mordredcommunity":
            lines.append(PY313_MORDREDCOMMUNITY_OVERRIDES["mordredcommunity"])
        elif key == "numpy":
            lines.append(PY313_MORDREDCOMMUNITY_OVERRIDES["numpy"])
        elif key == "drfp":
            lines.append(PY313_MORDREDCOMMUNITY_OVERRIDES["drfp"])
        else:
            lines.append(dist)

    return _dedupe_sort_req_lines(lines)


def write_py313_mordredcommunity_outputs(result: Dict[str, object], out_dir: Path) -> None:
    """Write the clean Python 3.13 + mordredcommunity profile outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    req_lines = _community_profile_lines(result)
    _write_plain_req(out_dir / "requirements-all-py313-mordredcommunity.txt", req_lines)

    install_md = [
        "# XTQSPRKinetic Python 3.13 clean installation with mordredcommunity",
        "",
        "This is the recommended Python 3.13 profile for XTQSPRKinetic v2.7.0 when",
        "using Mordred descriptors. It uses the community-maintained Mordred package",
        "while preserving the same Python import path used by the code:",
        "",
        "```python",
        "from mordred import Calculator, descriptors",
        "```",
        "",
        "## Generated requirement file",
        "",
        "```text",
        "requirements-all-py313-mordredcommunity.txt",
        "```",
        "",
        "The profile intentionally uses:",
        "",
        "```text",
        "mordredcommunity",
        "numpy>=2.2.2",
        "drfp==0.3.7",
        "```",
        "",
        "This avoids the NumPy 1.x conflict produced by the original Mordred 1.2.0",
        "compatibility override and should allow `python -m pip check` to pass in a",
        "clean Python 3.13 environment, subject to the rest of your platform wheels.",
        "",
        "## Recommended commands",
        "",
        "```bash",
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install -r requirements-all-py313-mordredcommunity.txt",
        "python -m pip check",
        "```",
        "",
        "## Runtime validation",
        "",
        "```bash",
        "python - << 'PY'",
        "import numpy, pandas, sklearn, xgboost, shap",
        "from mordred import Calculator, descriptors",
        "from rdkit import Chem",
        "print('numpy', numpy.__version__)",
        "print('mordred descriptors', len(Calculator(descriptors, ignore_3D=True).descriptors))",
        "print('rdkit molecule OK', Chem.MolFromSmiles('CC') is not None)",
        "PY",
        "```",
        "",
        "## Scientific validation note",
        "",
        "Before final manuscript/release use, compare a small descriptor table from the",
        "original Mordred environment and the mordredcommunity environment for a fixed",
        "set of reaction SMILES. If descriptor names and values are consistent within",
        "acceptable numerical tolerance, no XTQSPRKinetic source-code change is needed.",
        "",
    ]
    (out_dir / "INSTALL_PY313_MORDREDCOMMUNITY.md").write_text("\n".join(install_md), encoding="utf-8")


def write_py313_original_mordred_outputs(result: Dict[str, object], out_dir: Path) -> None:
    """Write the staged legacy Python 3.13 + original Mordred 1.2.0 outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dists = {str(x) for x in result["all_distributions"]}
    all_dists.update(PY313_BROAD_EXTRA_DISTS)

    broad = [
        d for d in sorted(all_dists, key=str.lower)
        if d.lower() not in {x.lower() for x in PY313_ORIGINAL_BROAD_EXCLUDE_DISTS}
    ]

    _write_plain_req(out_dir / "requirements-bootstrap-py313.txt", PY313_BOOTSTRAP_REQS)
    _write_plain_req(out_dir / "requirements-all-update-py313-original-mordred.txt", broad)
    _write_plain_req(out_dir / "requirements-runtime-py313-original-mordred.txt", PY313_ORIGINAL_MORDRED_RUNTIME_REQS)

    install_md = [
        "# XTQSPRKinetic Python 3.13 legacy installation with original mordred==1.2.0",
        "",
        "This profile preserves the original Mordred package for compatibility testing",
        "or exact legacy reproducibility. It is not the preferred clean Python 3.13",
        "profile because drfp 0.3.7 may declare a NumPy >=2.2.2 requirement, while",
        "original Mordred 1.2.0 is commonly stabilized by forcing NumPy 1.26.4.",
        "",
        "## Generated files",
        "",
        "```text",
        "requirements-bootstrap-py313.txt",
        "requirements-all-update-py313-original-mordred.txt",
        "requirements-runtime-py313-original-mordred.txt",
        "```",
        "",
        "## Recommended commands",
        "",
        "```bash",
        "python -m pip install --upgrade -r requirements-bootstrap-py313.txt",
        "python -m pip install -r requirements-all-update-py313-original-mordred.txt",
        "python -m pip install --force-reinstall -r requirements-runtime-py313-original-mordred.txt",
        "```",
        "",
        "The final runtime file intentionally sets:",
        "",
        "```text",
        *PY313_ORIGINAL_MORDRED_RUNTIME_REQS,
        "```",
        "",
        "## Important pip-check warning",
        "",
        "This legacy profile can be runtime-working while still failing metadata checks.",
        "For example, `python -m pip check` may report that drfp requires a newer",
        "NumPy than 1.26.4. That is expected for this legacy override profile and is",
        "why the mordredcommunity profile is preferred for clean Python 3.13 installs.",
        "",
        "## Runtime validation",
        "",
        "```bash",
        "python - << 'PY'",
        "import numpy, pandas, sklearn, xgboost, shap",
        "from mordred import Calculator, descriptors",
        "from rdkit import Chem",
        "print('numpy', numpy.__version__)",
        "print('mordred descriptors', len(Calculator(descriptors, ignore_3D=True).descriptors))",
        "print('rdkit molecule OK', Chem.MolFromSmiles('CC') is not None)",
        "PY",
        "```",
        "",
    ]
    (out_dir / "INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md").write_text("\n".join(install_md), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _normalize_profile(profile: str) -> str:
    p = str(profile or "standard").strip().lower()
    if p == "py313-mordred":
        return "py313-both"
    if p == "all":
        return "py313-both"
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate XTQSPRKinetic requirements files from imports.")
    parser.add_argument("paths", nargs="+", help="Python file(s) or directories to scan.")
    parser.add_argument("--out-dir", default=".", help="Output directory for requirements files.")
    parser.add_argument(
        "--pin",
        choices=["none", "installed", "strict"],
        default="installed",
        help="Version pinning mode for standard requirements outputs.",
    )
    parser.add_argument(
        "--xtqspr-defaults",
        action="store_true",
        help="Force-add known XTQSPRKinetic v2.7.0 core and optional packages.",
    )
    parser.add_argument(
        "--profile",
        choices=[
            "standard",
            "py313-mordredcommunity",
            "py313-original-mordred",
            "py313-both",
            "py313-mordred",  # backward-compatible alias for py313-both
            "all",            # alias for py313-both
        ],
        default="standard",
        help=(
            "Dependency-output profile. Use py313-mordredcommunity for a clean "
            "Python 3.13 + mordredcommunity profile, py313-original-mordred for "
            "legacy original Mordred 1.2.0, or py313-both to write both."
        ),
    )

    args = parser.parse_args(argv)
    profile = _normalize_profile(args.profile)

    paths = [Path(x).expanduser().resolve() for x in args.paths]
    result = scan(paths, xtqspr_defaults=bool(args.xtqspr_defaults))
    out_dir = Path(args.out_dir).expanduser().resolve()

    write_standard_outputs(result, out_dir, args.pin)

    profile_files: List[str] = []
    if profile in {"py313-mordredcommunity", "py313-both"}:
        write_py313_mordredcommunity_outputs(result, out_dir)
        profile_files.extend([
            "requirements-all-py313-mordredcommunity.txt",
            "INSTALL_PY313_MORDREDCOMMUNITY.md",
        ])

    if profile in {"py313-original-mordred", "py313-both"}:
        write_py313_original_mordred_outputs(result, out_dir)
        profile_files.extend([
            "requirements-bootstrap-py313.txt",
            "requirements-all-update-py313-original-mordred.txt",
            "requirements-runtime-py313-original-mordred.txt",
            "INSTALL_PY313_ORIGINAL_MORDRED_LEGACY.md",
        ])

    print(f"Scanned {len(result['files_scanned'])} Python files")
    print(f"Wrote requirements files to: {out_dir}")
    print(f"Core: {len(result['core_distributions'])} | Optional: {len(result['optional_distributions'])}")
    print(f"Profile: {args.profile} -> {profile}")
    if profile_files:
        print("Profile files:")
        for name in profile_files:
            print(f"  - {name}")
    if result["unresolved_import_modules"]:
        print("Unresolved imports:", ", ".join(result["unresolved_import_modules"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
