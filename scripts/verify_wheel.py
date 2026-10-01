"""Build from sdist and test an installed wheel with isolated Python."""

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="erabu-wheel-") as directory:
        work = Path(directory)
        artifacts = work / "artifacts"
        run("uv", "build", "--out-dir", str(artifacts))
        (sdist,) = artifacts.glob("*.tar.gz")
        source = work / "source"
        with tarfile.open(sdist) as archive:
            assert all("docs_local" not in name for name in archive.getnames())
            archive.extractall(source, filter="data")
        (project,) = source.iterdir()
        rebuilt = work / "rebuilt"
        run("uv", "build", str(sdist), "--wheel", "--out-dir", str(rebuilt))
        (wheel,) = rebuilt.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            assert "erabu/py.typed" in archive.namelist()
            assert all("docs_local" not in name for name in archive.namelist())
        requirements = work / "runtime.txt"
        run(
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--output-file",
            str(requirements),
            "--quiet",
        )
        environment = work / "consumer"
        run("uv", "venv", "--python", sys.executable, str(environment))
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--requirements",
            str(requirements),
            str(wheel),
        )
        check = (
            "from pathlib import Path; import erabu; "
            f"expected = Path({str(environment)!r}).resolve(); "
            "assert Path(erabu.__file__).resolve().is_relative_to(expected); "
            "print('Verified installed import:', erabu.__file__)"
        )
        run(str(python), "-I", "-c", check, cwd=work)
        for script in ("exercise.py", "shipping/main.py", "shifts.py"):
            run(str(python), "-I", str(project / "examples" / script), cwd=work)
    print("Source distribution, rebuilt wheel, and isolated examples passed.")


if __name__ == "__main__":
    main()
