#!/usr/bin/env python3
"""Fail closed unless the Multi-LoRA dependency provenance is exact."""

import argparse
import importlib.metadata
import json
import subprocess
import tempfile
from pathlib import Path

BRIDGE_REMOTE = "https://github.com/radixark/Megatron-Bridge"
BRIDGE_BRANCH = "bridge"
BRIDGE_SHA = "bb61fcd0b61f8acd0ef0f8b38b2240968c94c37b"
SGLANG_REMOTE = "https://github.com/sgl-project/sglang"
SGLANG_BRANCH = "sglang-miles"
MEGATRON_REMOTE = "https://github.com/radixark/Megatron-LM"
MEGATRON_BRANCH = "miles-main"


def _run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _normalize_remote(remote: str) -> str:
    remote = remote.strip().removesuffix(".git")
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    if remote.startswith("ssh://git@github.com/"):
        remote = "https://github.com/" + remote.removeprefix("ssh://git@github.com/")
    return remote


def _require_branch_contains(remote: str, branch: str, sha: str) -> str:
    with tempfile.TemporaryDirectory(prefix="miles-provenance-") as tmp:
        repo = Path(tmp)
        _run("git", "init", "--quiet", cwd=repo)
        _run("git", "fetch", "--quiet", "--filter=blob:none", "--no-tags", remote, branch, cwd=repo)
        branch_sha = _run("git", "rev-parse", "FETCH_HEAD", cwd=repo)
        if sha != branch_sha:
            _run("git", "fetch", "--quiet", "--filter=blob:none", "--no-tags", remote, sha, cwd=repo)
        _run("git", "merge-base", "--is-ancestor", sha, branch_sha, cwd=repo)
        return branch_sha


def _git_source(path: Path, remote: str, branch: str) -> dict[str, str]:
    actual_remote = _normalize_remote(_run("git", "remote", "get-url", "origin", cwd=path))
    if actual_remote != remote:
        raise RuntimeError(f"{path}: expected origin {remote}, got {actual_remote}")
    sha = _run("git", "rev-parse", "HEAD", cwd=path)
    branch_sha = _run("git", "rev-parse", f"refs/remotes/origin/{branch}", cwd=path)
    _run("git", "merge-base", "--is-ancestor", sha, branch_sha, cwd=path)
    return {"remote": remote, "branch": branch, "sha": sha, "branch_sha": branch_sha}


def _bridge_source() -> dict[str, str]:
    direct_url = importlib.metadata.distribution("megatron-bridge").read_text("direct_url.json")
    if not direct_url:
        raise RuntimeError("megatron-bridge lacks VCS provenance")
    source = json.loads(direct_url)
    remote = _normalize_remote(source["url"])
    sha = source.get("vcs_info", {}).get("commit_id")
    revision = source.get("vcs_info", {}).get("requested_revision")
    if (remote, sha, revision) != (BRIDGE_REMOTE, BRIDGE_SHA, BRIDGE_SHA):
        raise RuntimeError(f"unexpected megatron-bridge provenance: {remote}@{revision} ({sha})")
    branch_sha = _require_branch_contains(remote, BRIDGE_BRANCH, sha)
    return {"remote": remote, "branch": BRIDGE_BRANCH, "sha": sha, "branch_sha": branch_sha}


def _grouped_gemm_smoke() -> None:
    import torch
    from megatron.bridge.peft.multi_lora_layers import _apply_multi_lora_projection, _can_use_grouped_mm

    if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("the grouped-GEMM smoke requires a CUDA GPU and torch._grouped_mm")
    counts = (8, 8)
    inputs = torch.randn(sum(counts), 16, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(2, 16, 16, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    if not _can_use_grouped_mm(inputs, weights):
        raise RuntimeError("aligned Multi-LoRA tensors did not select grouped GEMM")
    actual = _apply_multi_lora_projection(inputs, weights, offsets, counts)
    expected = torch.cat(
        [torch.nn.functional.linear(x, w) for x, w in zip(inputs.split(counts), weights, strict=True)]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = {
        "megatron_bridge": _bridge_source(),
        "sglang": _git_source(Path("/sgl-workspace/sglang"), SGLANG_REMOTE, SGLANG_BRANCH),
        "megatron_lm": _git_source(Path("/root/Megatron-LM"), MEGATRON_REMOTE, MEGATRON_BRANCH),
    }
    if args.smoke:
        _grouped_gemm_smoke()
        result["grouped_gemm_smoke"] = "passed"
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
