"""Check public text, Markdown links and examples without external services."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "mise.toml",
    ".gitignore",
    ".python-version",
    "uv.lock",
)


def main() -> None:
    files = [ROOT / name for name in ROOT_FILES]
    for directory in ("src", "tests", "examples", "scripts", "benchmarks", "docs", ".github"):
        files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in (".py", ".md", ".toml", ".yml", ".typed")
        )
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.isascii(), f"Use English ASCII text outside docs_local: {path}"
        assert all(line == line.rstrip() for line in text.splitlines()), f"Trailing space: {path}"
        if path.suffix != ".md":
            continue
        fences = re.findall(r"^```([^\n]*)\n(.*?)^```\s*$", text, re.M | re.S)
        assert len(re.findall(r"^```", text, re.M)) == 2 * len(fences), path
        for language, source in fences:
            if language == "python":
                ast.parse(source, filename=str(path))
                if path.name == "README.md":
                    exec(compile(source, str(path), "exec"), {"__name__": "__documentation__"})
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            local = (path.parent / target.split("#", 1)[0]).resolve()
            assert local.is_file(), f"Broken link in {path}: {target}"
            assert "docs_local" not in local.parts, f"Private plan linked from {path}"
    print(f"Documentation and English-text checks passed ({len(files)} files).")


if __name__ == "__main__":
    main()
