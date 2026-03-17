from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any


INTERESTING_ENV_VARS = [
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "SLURM_JOB_ID",
    "SLURM_CPUS_ON_NODE",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS",
]


def _run_command(cmd: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not found"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    text = _read_text("/proc/meminfo")
    if text is None:
        return result
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        number = value.strip().split()[0]
        try:
            result[key] = int(number)
        except ValueError:
            continue
    return result


def _bytes_from_kib(value_kib: int | None) -> int | None:
    if value_kib is None:
        return None
    return value_kib * 1024


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    scaled = float(value)
    unit = units[0]
    for unit in units:
        if scaled < 1024.0 or unit == units[-1]:
            break
        scaled /= 1024.0
    return f"{scaled:.2f} {unit}"


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": None,
        "affinity_list": None,
        "taskset_available": shutil.which("taskset") is not None,
        "numactl_available": shutil.which("numactl") is not None,
    }
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        info["affinity_cpus"] = len(affinity)
        info["affinity_list"] = affinity

    rc, stdout, _ = _run_command(["lscpu"])
    if rc == 0:
        info["lscpu"] = stdout
    return info


def _memory_info() -> dict[str, Any]:
    meminfo = _parse_meminfo()
    return {
        "mem_total_bytes": _bytes_from_kib(meminfo.get("MemTotal")),
        "mem_available_bytes": _bytes_from_kib(meminfo.get("MemAvailable")),
        "swap_total_bytes": _bytes_from_kib(meminfo.get("SwapTotal")),
        "swap_free_bytes": _bytes_from_kib(meminfo.get("SwapFree")),
    }


def _numa_info() -> dict[str, Any]:
    info: dict[str, Any] = {"nodes": []}
    rc, stdout, stderr = _run_command(["numactl", "--hardware"])
    if rc == 0:
        info["numactl_hardware"] = stdout
        return info
    if stderr:
        info["numactl_hardware_error"] = stderr

    sysfs_root = Path("/sys/devices/system/node")
    if not sysfs_root.exists():
        return info
    for path in sorted(sysfs_root.glob("node[0-9]*")):
        cpulist = _read_text(str(path / "cpulist"))
        meminfo = _read_text(str(path / "meminfo"))
        info["nodes"].append(
            {
                "node": path.name,
                "cpulist": cpulist,
                "meminfo": meminfo,
            }
        )
    return info


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpus": []}
    query_fields = [
        "index",
        "name",
        "memory.total",
        "memory.free",
        "utilization.gpu",
    ]
    rc, stdout, stderr = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(query_fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc == 0 and stdout:
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != len(query_fields):
                continue
            index, name, mem_total, mem_free, util = parts
            info["gpus"].append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mib": int(mem_total),
                    "memory_free_mib": int(mem_free),
                    "utilization_gpu_percent": int(util),
                }
            )
    elif stderr:
        info["nvidia_smi_error"] = stderr

    rc, stdout, stderr = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc == 0 and stdout:
        processes = []
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            gpu_uuid, pid, process_name, used_memory = parts
            processes.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mib": int(used_memory),
                }
            )
        info["compute_processes"] = processes
    elif stderr:
        info["compute_processes_error"] = stderr
    return info


def _python_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_executable": shutil.which("python"),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": torch.cuda.get_device_capability(idx),
                }
                for idx in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def _env_info() -> dict[str, Any]:
    return {key: os.environ.get(key) for key in INTERESTING_ENV_VARS if os.environ.get(key) is not None}


def _suggestions(payload: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    cpu_info = payload["cpu"]
    memory_info = payload["memory"]
    gpu_info = payload["gpu"]

    affinity_cpus = cpu_info.get("affinity_cpus")
    if affinity_cpus:
        thread_budget = max(1, min(affinity_cpus, 8))
        suggestions.append(
            f"CPU-only pinning example: OMP_NUM_THREADS={thread_budget} MKL_NUM_THREADS={thread_budget} "
            f"OPENBLAS_NUM_THREADS={thread_budget} taskset -c 0-{thread_budget - 1} python ..."
        )

    if cpu_info.get("numactl_available"):
        suggestions.append("NUMA pinning example: numactl --cpunodebind=0 --membind=0 python ...")

    if gpu_info["gpus"]:
        first_gpu = gpu_info["gpus"][0]["index"]
        suggestions.append(f"Single-GPU pinning example: CUDA_VISIBLE_DEVICES={first_gpu} python ...")
        if len(gpu_info["gpus"]) >= 2:
            suggestions.append("Two-GPU pinning example: CUDA_VISIBLE_DEVICES=0,1 python ...")

    mem_available = memory_info.get("mem_available_bytes")
    if mem_available is not None:
        suggestions.append(f"Current available RAM: {_format_bytes(mem_available)}")

    suggestions.append(
        "For this project, fix both GPU and CPU resources together when benchmarking, "
        "otherwise solver timings will be noisy."
    )
    return suggestions


def collect_payload() -> dict[str, Any]:
    hostname = socket.gethostname()
    payload = {
        "hostname": hostname,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "numa": _numa_info(),
        "gpu": _gpu_info(),
        "python": _python_info(),
        "env": _env_info(),
    }
    payload["suggestions"] = _suggestions(payload)
    return payload


def _render_human(payload: dict[str, Any]) -> str:
    lines = []
    lines.append(f"host: {payload['hostname']}")
    lines.append(
        "platform: "
        f"{payload['platform']['system']} {payload['platform']['release']} "
        f"({payload['platform']['machine']}) Python {payload['platform']['python_version']}"
    )

    cpu = payload["cpu"]
    lines.append(
        f"cpu: logical={cpu.get('logical_cpus')} affinity={cpu.get('affinity_cpus')} "
        f"taskset={cpu.get('taskset_available')} numactl={cpu.get('numactl_available')}"
    )
    if cpu.get("affinity_list"):
        lines.append(f"cpu_affinity_list: {cpu['affinity_list']}")

    memory = payload["memory"]
    lines.append(
        f"memory: total={_format_bytes(memory.get('mem_total_bytes'))} "
        f"available={_format_bytes(memory.get('mem_available_bytes'))} "
        f"swap_total={_format_bytes(memory.get('swap_total_bytes'))}"
    )

    gpus = payload["gpu"]["gpus"]
    lines.append(f"gpus: count={len(gpus)}")
    for gpu in gpus:
        lines.append(
            f"gpu[{gpu['index']}]: {gpu['name']} total={gpu['memory_total_mib']} MiB "
            f"free={gpu['memory_free_mib']} MiB util={gpu['utilization_gpu_percent']}%"
        )

    if payload["env"]:
        lines.append("env:")
        for key, value in payload["env"].items():
            lines.append(f"  {key}={value}")

    lines.append("suggestions:")
    for suggestion in payload["suggestions"]:
        lines.append(f"  - {suggestion}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect server CPU/GPU resources and pinning options.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--human-only", action="store_true")
    args = parser.parse_args()

    payload = collect_payload()
    human = _render_human(payload)
    print(human)

    if not args.human_only:
        print()
        print(json.dumps(payload, indent=2))

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
