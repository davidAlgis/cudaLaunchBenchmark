#!/usr/bin/env python3
"""
setup.py  (formerly cmake_build.py)

Configure, build, and run a CMake project even when this script
lives in a subfolder (e.g., scripts/). It auto-detects the repo root
by searching upward for CMakeLists.txt, unless --source-dir is given.

Usage examples:
  python scripts/setup.py configure
  python scripts/setup.py build --config Debug
  python scripts/setup.py run --config Release
  python scripts/setup.py all --config Release -- -n 2000000 --iters 128
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Defaults you can tweak
DEFAULT_BUILD_DIR = "build"
DEFAULT_TARGET = "cudaLaunchBench"


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as exc:
        print(f"[error] command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)


def which_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def detect_default_generator() -> Optional[str]:
    if which_tool("ninja"):
        return "Ninja"
    return None


def find_project_root(start: Path) -> Path:
    """
    Walk upward from 'start' to find a directory containing CMakeLists.txt.
    If not found, default to the parent of this script (one level up).
    """
    for p in [start] + list(start.parents):
        if (p / "CMakeLists.txt").exists():
            return p
    # Fallback: scripts/.. (typical layout)
    fallback = start.parent
    print(
        f"[warn] CMakeLists.txt not found upward from {start}. "
        f"Falling back to: {fallback}"
    )
    return fallback


def cmake_configure(
    source_dir: Path,
    build_dir: Path,
    generator: Optional[str],
    extra_cmake_args: List[str],
    config: str,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if generator:
        cmd += ["-G", generator]
    if config:
        cmd += [f"-DCMAKE_BUILD_TYPE={config}"]
    cmd += extra_cmake_args
    run_cmd(cmd)


def cmake_build(build_dir: Path, config: str, parallel: Optional[int]) -> None:
    cmd = ["cmake", "--build", str(build_dir)]
    if config:
        cmd += ["--config", config]
    if parallel is not None and parallel > 0:
        if platform.system() == "Windows":
            cmd += ["--", f"/m:{parallel}"]
        else:
            cmd += ["--", f"-j{parallel}"]
    run_cmd(cmd)


def exe_candidates(build_dir: Path, target: str, config: str) -> List[Path]:
    exe_name = f"{target}.exe" if platform.system() == "Windows" else target
    candidates = [
        build_dir / exe_name,
        build_dir / "bin" / exe_name,
        build_dir / config / exe_name,
        build_dir / "src" / config / exe_name,
        build_dir / "apps" / config / exe_name,
        build_dir / "Release" / exe_name,
        build_dir / "Debug" / exe_name,
    ]
    seen, unique = set(), []
    for p in candidates:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def find_executable(build_dir: Path, target: str, config: str) -> Path:
    for cand in exe_candidates(build_dir, target, config):
        if cand.exists():
            return cand
    return exe_candidates(build_dir, target, config)[0]


def run_executable(exe: Path, run_args: List[str]) -> int:
    if not exe.exists():
        print(f"[warn] executable not found at: {exe}")
    cmd = [str(exe)] + run_args
    print(f"[run] {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("[error] failed to start process (file not found)")
        return 127


def parse_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Configure, build, and run a CMake project."
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--source-dir",
            default=None,
            help="Path to project root containing CMakeLists.txt. "
            "Default: auto-detect by walking up from this script.",
        )
        p.add_argument(
            "--build-dir",
            default=DEFAULT_BUILD_DIR,
            help="Build directory (created inside source-dir unless absolute). "
            f"(default: {DEFAULT_BUILD_DIR})",
        )
        p.add_argument(
            "--config",
            default="Debug",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration (default: Debug)",
        )
        p.add_argument(
            "--generator",
            default=None,
            help='CMake generator (e.g., "Ninja", "Visual Studio 17 2022"). '
            "Default: Ninja if available, else CMake default.",
        )
        p.add_argument(
            "--target",
            default=DEFAULT_TARGET,
            help=f"Target executable name (default: {DEFAULT_TARGET})",
        )
        p.add_argument(
            "--parallel",
            type=int,
            default=None,
            help="Parallel build jobs (e.g., 8). Default: tool default.",
        )
        p.add_argument(
            "--cmake",
            nargs=argparse.REMAINDER,
            default=[],
            help="Extra args for CMake configure (use after --cmake ...).",
        )

    add_common(sub.add_parser("configure", help="Run CMake configure"))
    add_common(sub.add_parser("build", help="Build the project"))
    add_common(sub.add_parser("run", help="Run the executable"))
    add_common(sub.add_parser("all", help="Configure, build, and run"))

    # Preserve anything after "--" as runtime args for the executable.
    if "--" in argv:
        idx = argv.index("--")
        known, rest = argv[:idx], argv[idx + 1 :]
    else:
        known, rest = argv, []

    args = parser.parse_args(known)
    return args, rest


def main() -> None:
    args, run_args = parse_args(sys.argv[1:])

    script_dir = Path(__file__).resolve().parent
    # Resolve source dir: explicit > auto-detected
    if args.source_dir:
        source_dir = Path(args.source_dir).resolve()
    else:
        source_dir = find_project_root(script_dir)

    # Build dir: relative to source_dir unless absolute provided
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (source_dir / build_dir).resolve()

    # Tool presence checks
    if not which_tool("cmake"):
        print("[error] cmake not found in PATH")
        sys.exit(127)

    generator = args.generator or detect_default_generator()
    extra_cmake_args = args.cmake or []

    print(f"[info] source-dir = {source_dir}")
    print(f"[info] build-dir  = {build_dir}")
    if generator:
        print(f"[info] generator  = {generator}")

    if args.action == "configure":
        cmake_configure(
            source_dir, build_dir, generator, extra_cmake_args, args.config
        )
        return

    if args.action == "build":
        if not (build_dir / "CMakeCache.txt").exists():
            print(
                "[info] build dir not configured yet, running configure first..."
            )
            cmake_configure(
                source_dir, build_dir, generator, extra_cmake_args, args.config
            )
        cmake_build(build_dir, args.config, args.parallel)
        return

    if args.action == "run":
        exe = find_executable(build_dir, args.target, args.config)
        code = run_executable(exe, run_args)
        sys.exit(code)

    if args.action == "all":
        cmake_configure(
            source_dir, build_dir, generator, extra_cmake_args, args.config
        )
        cmake_build(build_dir, args.config, args.parallel)
        exe = find_executable(build_dir, args.target, args.config)
        code = run_executable(exe, run_args)
        sys.exit(code)


if __name__ == "__main__":
    main()
