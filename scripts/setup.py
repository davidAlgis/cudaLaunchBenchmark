#!/usr/bin/env python3
"""
setup.py — configure, build, run (with color + UTF-8 friendly output)

- Auto-finds project root (CMakeLists.txt) from scripts/...
- On Windows, wires CUDA toolset and forces MSBuild colored output.
- Forces UTF-8 for Python and common tools.
- Lets you choose config, arch, etc.

Usage:
  py -3 scripts\setup.py configure
  py -3 scripts\setup.py build --config Release
  py -3 scripts\setup.py run --config Release -- -n 2000000 --iters 128
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_BUILD_DIR = "build"
DEFAULT_TARGET = "cudaLaunchBench"
CUDA_DEFAULT_PATH_WIN = (
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
)
DEFAULT_CUDA_ARCH = "75"  # override with --cuda-arch

# ---------------------------- helpers --------------------------------


def which_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def run_cmd(
    cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None
) -> int:
    """
    Run a command streaming stdout/stderr to parent (keeps tool colors if enabled).
    Returns exit code.
    """
    print(f"[cmd] {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env)
        return proc.wait()
    except FileNotFoundError:
        print(f"[error] command not found: {cmd[0]}")
        return 127


def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "CMakeLists.txt").exists():
            return p
    fallback = start.parent
    print(
        f"[warn] CMakeLists.txt not found upward from {start}. Falling back to: {fallback}"
    )
    return fallback


def prepare_utf8_color_env(base: Optional[dict] = None) -> dict:
    """
    Ensure UTF-8 and ANSI colors are encouraged even without a TTY.
    """
    env = dict(base or os.environ)
    # Unicode/UTF-8
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault(
        "LC_ALL", "C.UTF-8" if platform.system() != "Windows" else ""
    )
    # ANSI color nudges commonly honored by many tools
    env.setdefault("CLICOLOR", "1")
    env.setdefault("CLICOLOR_FORCE", "1")
    env.setdefault("FORCE_COLOR", "3")
    env.setdefault(
        "TERM", env.get("TERM", "xterm-256color") or "xterm-256color"
    )
    return env


def prepare_cuda_on_windows(
    cuda_path_opt: Optional[str], env: dict
) -> Tuple[dict, List[str]]:
    """
    Prepare CUDA env and extra CMake args for Windows VS generators.
    Returns (env, extra_cmake_args).
    """
    extra: List[str] = []
    cuda_path = Path(cuda_path_opt or CUDA_DEFAULT_PATH_WIN)
    if not cuda_path.exists():
        print(f"[warn] CUDA path does not exist: {cuda_path}")
        return env, extra

    nvcc = cuda_path / "bin" / "nvcc.exe"
    if not nvcc.exists():
        print(f"[warn] nvcc not found at: {nvcc}")
        return env, extra

    # Env
    env["CUDA_PATH"] = str(cuda_path)
    env["CUDACXX"] = str(nvcc)
    env["PATH"] = str(cuda_path / "bin") + os.pathsep + env.get("PATH", "")

    # CMake toolset + CUDA compiler hint
    extra.append(f"-DCMAKE_GENERATOR_TOOLSET=cuda={cuda_path}")
    extra.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

    print(f"[info] Using CUDA at: {cuda_path}")
    return env, extra


# ---------------------------- CMake steps -----------------------------


def cmake_configure(
    source_dir: Path,
    build_dir: Path,
    config: str,
    cuda_arch: Optional[str],
    extra_cmake_args: List[str],
    env: dict,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if config:
        cmd += [f"-DCMAKE_BUILD_TYPE={config}"]
    if cuda_arch:
        cmd += [f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
    # Encourage colored diagnostics (Clang/GCC)
    cmd += ["-DCMAKE_COLOR_DIAGNOSTICS=ON"]
    cmd += extra_cmake_args
    code = run_cmd(cmd, env=env)
    if code != 0:
        sys.exit(code)


def cmake_build(
    build_dir: Path, config: str, parallel: Optional[int], env: dict
) -> None:
    cmd = ["cmake", "--build", str(build_dir)]
    if config:
        cmd += ["--config", config]

    # Append tool-specific color/parallel flags
    if platform.system() == "Windows":
        # MSBuild colored output
        msbuild_color = "/consoleloggerparameters:ForceConsoleColor;WarningColor=Cyan;ErrorColor=Yellow"
        if parallel and parallel > 0:
            cmd += ["--", f"/m:{parallel}", msbuild_color]
        else:
            cmd += ["--", msbuild_color]
    else:
        # Ninja/Make: -j plus many tools honor CLICOLOR/TERM already
        if parallel and parallel > 0:
            cmd += ["--", f"-j{parallel}"]

    code = run_cmd(cmd, env=env)
    if code != 0:
        sys.exit(code)


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


def run_executable(exe: Path, run_args: List[str], env: dict) -> int:
    if not exe.exists():
        print(f"[warn] executable not found at: {exe}")
    cmd = [str(exe)] + run_args
    print(f"[run] {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, env=env)
        return proc.wait()
    except FileNotFoundError:
        print("[error] failed to start process (file not found)")
        return 127


# ---------------------------- CLI ------------------------------------


def parse_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Configure, build, and run a CMake project."
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--source-dir",
            default=None,
            help="Path to project root containing CMakeLists.txt. Default: auto-detect.",
        )
        p.add_argument(
            "--build-dir",
            default=DEFAULT_BUILD_DIR,
            help=f"Build directory (relative to source-dir). Default: {DEFAULT_BUILD_DIR}",
        )
        p.add_argument(
            "--config",
            default="Debug",
            choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"],
            help="Build configuration.",
        )
        p.add_argument(
            "--target",
            default=DEFAULT_TARGET,
            help=f"Target executable name. Default: {DEFAULT_TARGET}",
        )
        p.add_argument(
            "--parallel",
            type=int,
            default=None,
            help="Parallel build jobs (e.g., 8).",
        )
        p.add_argument(
            "--cuda-path",
            default=None,
            help=(
                "CUDA toolkit root on Windows, e.g. "
                r'"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3".'
            ),
        )
        p.add_argument(
            "--cuda-arch",
            default=DEFAULT_CUDA_ARCH,
            help=f"Value for CMAKE_CUDA_ARCHITECTURES (default: {DEFAULT_CUDA_ARCH}).",
        )
        p.add_argument(
            "--export-compile-commands",
            action="store_true",
            help="Pass -DCMAKE_EXPORT_COMPILE_COMMANDS=ON at configure.",
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


# ---------------------------- main -----------------------------------


def main() -> None:
    args, run_args = parse_args(sys.argv[1:])

    script_dir = Path(__file__).resolve().parent
    source_dir = (
        Path(args.source_dir).resolve()
        if args.source_dir
        else find_project_root(script_dir)
    )

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (source_dir / build_dir).resolve()

    if not which_tool("cmake"):
        print("[error] cmake not found in PATH")
        sys.exit(127)

    # Base env: UTF-8 + color
    env = prepare_utf8_color_env()

    # Export compile_commands.json for tooling, if asked
    extra_cmake_args: List[str] = []
    if args.export_compile_commands:
        extra_cmake_args.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

    # CUDA wiring on Windows (same pattern as your other project)
    if platform.system() == "Windows":
        env, cuda_cmake_args = prepare_cuda_on_windows(args.cuda_path, env)
        extra_cmake_args += cuda_cmake_args

    # Pass along user-provided extra cmake args after --cmake ...
    extra_cmake_args += args.cmake or []

    if args.action == "configure":
        cmake_configure(
            source_dir,
            build_dir,
            args.config,
            args.cuda_arch,
            extra_cmake_args,
            env,
        )
        return

    if args.action == "build":
        if not (build_dir / "CMakeCache.txt").exists():
            print(
                "[info] build dir not configured yet, running configure first..."
            )
            cmake_configure(
                source_dir,
                build_dir,
                args.config,
                args.cuda_arch,
                extra_cmake_args,
                env,
            )
        cmake_build(build_dir, args.config, args.parallel, env)
        return

    if args.action == "run":
        exe = find_executable(build_dir, args.target, args.config)
        code = run_executable(exe, run_args, env)
        sys.exit(code)

    if args.action == "all":
        cmake_configure(
            source_dir,
            build_dir,
            args.config,
            args.cuda_arch,
            extra_cmake_args,
            env,
        )
        cmake_build(build_dir, args.config, args.parallel, env)
        exe = find_executable(build_dir, args.target, args.config)
        code = run_executable(exe, run_args, env)
        sys.exit(code)


if __name__ == "__main__":
    main()
