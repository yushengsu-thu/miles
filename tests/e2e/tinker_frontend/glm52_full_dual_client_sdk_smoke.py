#!/usr/bin/env python3
"""Mechanical dual-client smoke for full GLM-5.2 through the public Tinker API.

The unmodified official ``tinker==0.24.1`` SDK creates two independent LoRA
training clients (logical ranks 8 and 16). For forward-backward, optimizer
step, forward, and sampling, both requests are submitted before either result
is awaited. This gives the backend a real two-adapter batching opportunity.
The clients train on one shared prompt with different completion-only targets,
then both published samplers generate ordinary completions. Each completion is
scored by both training clients at the same BF16 precision and must be more
likely under its originating client, catching swapped or aliased serving
adapters without comparing BF16 trainer and FP8 serving logprobs directly.

This is deliberately a small interface/mechanics gate. It does not send
routing replay, compute rewards or advantages, or claim on-policy RL quality.
It imports no Miles or Ray API.

The pinned SDK has no public sampling-session close call. The script unloads
both training models and frees their slots, but the disposable acceptance
deployment should still be stopped after the gate.

Run from the Ray head against an already-running GLM-5.2 Tinker deployment::

    python tests/e2e/tinker_frontend/glm52_full_dual_client_sdk_smoke.py \
      --base-url http://127.0.0.1:8068 \
      --base-model /cluster-storage/models/GLM-5.2 \
      --out-dir /scratch/glm52-tinker-sdk-smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import tinker
from tinker import types

SDK_VERSION = "0.24.1"
ADAPTER_RANKS = {"rank8": 8, "rank16": 16}
TARGET_TOKENS = {"rank8": 1200, "rank16": 2200}
SAMPLE_TOKEN_BASE = 3200


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8068")
    parser.add_argument(
        "--base-model",
        help="exact advertised model name; default: discover the sole model",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MILES_TINKER_API_KEY", "tml-miles-gpu-acceptance"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=_positive_float, default=1800.0)
    parser.add_argument(
        "--datum-tokens",
        type=_positive_int,
        default=8,
        help="target positions per training datum",
    )
    parser.add_argument("--train-steps", type=_positive_int, default=2)
    parser.add_argument("--sample-prompt-tokens", type=_positive_int, default=8)
    parser.add_argument("--sample-max-tokens", type=_positive_int, default=4)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _await_pair(
    phase: str, pending: dict[str, Any], timeout_s: float
) -> tuple[dict[str, Any], float]:
    """Await a pair only after both SDK futures have already been submitted."""
    if set(pending) != set(ADAPTER_RANKS):
        raise AssertionError(
            f"{phase}: expected futures for {list(ADAPTER_RANKS)}, got {list(pending)}"
        )

    started = time.monotonic()
    deadline = started + timeout_s
    results: dict[str, Any] = {}
    errors: dict[str, Exception] = {}
    _log(f"{phase}: both futures submitted; entering central barrier")
    for name, future in pending.items():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            results[name] = future.result(timeout=remaining)
        except Exception as exc:  # noqa: BLE001 - report both clients after the barrier
            errors[name] = exc

    elapsed = time.monotonic() - started
    if errors:
        details = ", ".join(
            f"{name}={type(exc).__name__}: {exc}" for name, exc in errors.items()
        )
        raise RuntimeError(f"{phase} failed after {elapsed:.1f}s: {details}") from next(
            iter(errors.values())
        )
    _log(f"{phase}: both results completed in {elapsed:.1f}s")
    return results, elapsed


def _ce_datum_from_tokens(tokens: list[int]) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )


def _completion_ce_datum(
    prompt_tokens: list[int], target_token: int, target_positions: int
) -> types.Datum:
    completion_tokens = [target_token] * target_positions
    tokens = prompt_tokens + completion_tokens
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [0.0] * (len(prompt_tokens) - 1) + [1.0] * target_positions,
        },
    )


def _check_forward_result(
    name: str, result: Any, target_positions: int
) -> dict[str, Any]:
    if len(result.loss_fn_outputs) != 1:
        raise AssertionError(
            f"{name}: expected one loss output, got {len(result.loss_fn_outputs)}"
        )
    row = [float(item) for item in result.loss_fn_outputs[0]["logprobs"].tolist()]
    if len(row) != target_positions or not all(math.isfinite(item) for item in row):
        raise AssertionError(f"{name}: malformed logprobs: {row}")
    loss_sum = float(result.metrics["loss:sum"])
    if not math.isfinite(loss_sum):
        raise AssertionError(f"{name}: non-finite loss:sum={loss_sum}")
    return {"loss_sum": loss_sum, "logprobs": row}


def _check_optim_result(name: str, result: Any) -> float:
    grad_norm = float(result.metrics["grad_norm"])
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError(
            f"{name}: expected a positive finite grad_norm, got {grad_norm}"
        )
    return grad_norm


def _check_sample_result(name: str, response: Any, max_tokens: int) -> dict[str, Any]:
    if len(response.sequences) != 1:
        raise AssertionError(
            f"{name}: expected one sampled sequence, got {len(response.sequences)}"
        )
    sequence = response.sequences[0]
    tokens = [int(token) for token in sequence.tokens]
    logprobs = [float(item) for item in (sequence.logprobs or [])]
    if not tokens or len(tokens) > max_tokens:
        raise AssertionError(f"{name}: unexpected sample token count: {len(tokens)}")
    if len(logprobs) != len(tokens) or not all(
        math.isfinite(item) for item in logprobs
    ):
        raise AssertionError(f"{name}: malformed sample logprobs: {logprobs}")
    return {
        "tokens": tokens,
        "logprobs": logprobs,
        "stop_reason": str(sequence.stop_reason),
    }


def _assert_parameter_update(
    name: str, before: dict[str, Any], after: dict[str, Any]
) -> float:
    before_logprobs = before["logprobs"]
    after_logprobs = after["logprobs"]
    if len(before_logprobs) != len(after_logprobs):
        raise AssertionError(f"{name}: pre/post forward lengths differ")
    max_delta = max(
        abs(left - right)
        for left, right in zip(before_logprobs, after_logprobs, strict=True)
    )
    if max_delta <= 1e-6:
        raise AssertionError(
            f"{name}: optimizer step did not measurably change same-datum logprobs"
        )
    return max_delta


def _mean_abs_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise AssertionError(
            f"cannot compare logprob rows of lengths {len(left)} and {len(right)}"
        )
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def _assert_published_association(
    samples: dict[str, dict[str, Any]],
    training_scores: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, other_name in (("rank8", "rank16"), ("rank16", "rank8")):
        sampled_logprobs = samples[name]["logprobs"]
        completion_tokens = len(samples[name]["tokens"])
        own_logprobs = training_scores[name][name]["logprobs"][-completion_tokens:]
        cross_logprobs = training_scores[name][other_name]["logprobs"][
            -completion_tokens:
        ]
        training_gap = max(
            abs(left - right)
            for left, right in zip(own_logprobs, cross_logprobs, strict=True)
        )
        if training_gap <= 1e-6:
            raise AssertionError(
                f"training adapters are indistinguishable on the {name} sampled completion"
            )
        own_score = sum(own_logprobs) / completion_tokens
        cross_score = sum(cross_logprobs) / completion_tokens
        score_margin = own_score - cross_score
        own_error = _mean_abs_error(sampled_logprobs, own_logprobs)
        cross_error = _mean_abs_error(sampled_logprobs, cross_logprobs)
        if score_margin <= 1e-4:
            raise AssertionError(
                f"published {name} sampler is not more likely under its originating training client: "
                f"own_score={own_score}, cross_score={cross_score}, score_margin={score_margin}, "
                f"serving_to_own_mae={own_error}, serving_to_cross_mae={cross_error}"
            )
        summary[name] = {
            "training_max_logprob_delta": training_gap,
            "own_score": own_score,
            "cross_score": cross_score,
            "association_score_margin": score_margin,
            "serving_to_own_mae": own_error,
            "serving_to_cross_mae": cross_error,
        }
    return summary


def _resolve_base_model(service: Any, requested: str | None) -> tuple[str, list[str]]:
    advertised = [
        model.model_name for model in service.get_server_capabilities().supported_models
    ]
    if requested is not None:
        if requested not in advertised:
            raise AssertionError(
                f"requested base model {requested!r} is not advertised: {advertised}"
            )
        return requested, advertised
    if len(advertised) != 1 or not advertised[0]:
        raise AssertionError(
            f"--base-model omitted, but the deployment advertises {advertised}"
        )
    return advertised[0], advertised


def _unload_model_ids(
    base_url: str, api_key: str, model_ids: list[Any], timeout_s: float
) -> None:
    async def unload_all() -> None:
        from tinker._client import AsyncTinker

        low_level = AsyncTinker(base_url=base_url, api_key=api_key)
        errors: dict[str, Exception] = {}
        try:
            for model_id in model_ids:
                label = str(model_id)
                try:
                    future = await low_level.models.unload(
                        request=types.UnloadModelRequest(model_id=model_id)
                    )
                    deadline = time.monotonic() + min(timeout_s, 120.0)
                    while time.monotonic() < deadline:
                        raw = await low_level.futures.with_raw_response.retrieve(
                            request=types.FutureRetrieveRequest(
                                request_id=future.request_id
                            )
                        )
                        body = await raw.json()
                        if body.get("type") == "try_again":
                            await asyncio.sleep(0.1)
                            continue
                        if body != {"type": "unload_model", "model_id": model_id}:
                            raise RuntimeError(f"unexpected unload response: {body}")
                        _log(f"unloaded training model {label}")
                        break
                    else:
                        raise TimeoutError(
                            f"unload did not finish within {min(timeout_s, 120.0):g}s"
                        )
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - try every model before reporting cleanup failure
                    errors[label] = exc
        finally:
            await low_level.close()
        if errors:
            details = ", ".join(
                f"{model_id}={type(exc).__name__}: {exc}"
                for model_id, exc in errors.items()
            )
            raise RuntimeError(f"model cleanup failed: {details}")

    asyncio.run(unload_all())


def _run_smoke(
    args: argparse.Namespace, clients: dict[str, Any], progress: dict[str, Any]
) -> dict[str, Any]:
    services = {
        name: tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
        for name in ADAPTER_RANKS
    }
    base_model, advertised = _resolve_base_model(services["rank8"], args.base_model)
    _log(f"server advertises {advertised}; using {base_model!r}")
    progress.update(
        {
            "sdk_version": version("tinker"),
            "base_url": args.base_url,
            "base_model": base_model,
            "advertised_models": advertised,
        }
    )

    for name, rank in ADAPTER_RANKS.items():
        clients[name] = services[name].create_lora_training_client(
            base_model=base_model, rank=rank
        )
    client_info: dict[str, dict[str, Any]] = {}
    for name, expected_rank in ADAPTER_RANKS.items():
        info = clients[name].get_info()
        if info.lora_rank != expected_rank:
            raise AssertionError(
                f"{name}: expected logical rank {expected_rank}, got {info.lora_rank}"
            )
        client_info[name] = {
            "model_id": str(clients[name].model_id),
            "lora_rank": info.lora_rank,
        }
    if len({client.model_id for client in clients.values()}) != len(clients):
        raise AssertionError(f"training clients are not distinct: {client_info}")
    _log(f"dual clients ready: {client_info}")
    progress["clients"] = client_info

    prompt_tokens = list(
        range(SAMPLE_TOKEN_BASE, SAMPLE_TOKEN_BASE + args.sample_prompt_tokens)
    )
    data = {
        name: [
            _completion_ce_datum(prompt_tokens, TARGET_TOKENS[name], args.datum_tokens)
        ]
        for name in ADAPTER_RANKS
    }
    training_positions = len(prompt_tokens) + args.datum_tokens - 1
    progress["training_objectives"] = {
        name: {
            "target_token": TARGET_TOKENS[name],
            "target_positions": args.datum_tokens,
        }
        for name in ADAPTER_RANKS
    }

    training_steps: list[dict[str, Any]] = []
    phase_seconds: dict[str, float] = {}
    for step in range(1, args.train_steps + 1):
        fb_phase = f"forward_backward_step_{step}"
        fb_results, fb_seconds = _await_pair(
            fb_phase,
            {
                name: clients[name].forward_backward(data[name], "cross_entropy")
                for name in ADAPTER_RANKS
            },
            args.timeout_s,
        )
        fb_summary = {
            name: _check_forward_result(name, fb_results[name], training_positions)
            for name in ADAPTER_RANKS
        }

        optim_phase = f"optim_step_{step}"
        optim_results, optim_seconds = _await_pair(
            optim_phase,
            {
                name: clients[name].optim_step(
                    types.AdamParams(learning_rate=args.learning_rate)
                )
                for name in ADAPTER_RANKS
            },
            args.timeout_s,
        )
        grad_norms = {
            name: _check_optim_result(name, optim_results[name])
            for name in ADAPTER_RANKS
        }
        training_steps.append(
            {"step": step, "forward_backward": fb_summary, "grad_norms": grad_norms}
        )
        phase_seconds[fb_phase] = round(fb_seconds, 3)
        phase_seconds[optim_phase] = round(optim_seconds, 3)
        progress["training_steps"] = training_steps

    forward_results, forward_seconds = _await_pair(
        "forward",
        {
            name: clients[name].forward(data[name], "cross_entropy")
            for name in ADAPTER_RANKS
        },
        args.timeout_s,
    )
    forward_summary = {
        name: _check_forward_result(name, forward_results[name], training_positions)
        for name in ADAPTER_RANKS
    }
    max_logprob_changes = {
        name: _assert_parameter_update(
            name, training_steps[0]["forward_backward"][name], forward_summary[name]
        )
        for name in ADAPTER_RANKS
    }
    progress["forward"] = forward_summary
    progress["max_logprob_changes"] = max_logprob_changes

    sampling_clients: dict[str, Any] = {}
    for name in ADAPTER_RANKS:
        _log(f"publish {name}: save_weights_and_get_sampling_client")
        sampling_clients[name] = clients[name].save_weights_and_get_sampling_client()
        if sampling_clients[name].get_base_model() != base_model:
            raise AssertionError(
                f"{name}: published sampler reports the wrong base model"
            )

    prompt = types.ModelInput.from_ints(prompt_tokens)
    sample_results, sample_seconds = _await_pair(
        "sample",
        {
            name: sampling_clients[name].sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=args.sample_max_tokens, temperature=0.0
                ),
            )
            for name in ADAPTER_RANKS
        },
        args.timeout_s,
    )
    samples = {
        name: _check_sample_result(name, sample_results[name], args.sample_max_tokens)
        for name in ADAPTER_RANKS
    }
    progress["samples"] = samples

    association_forward: dict[str, dict[str, dict[str, Any]]] = {}
    association_forward_seconds: dict[str, float] = {}
    for sampled_name in ADAPTER_RANKS:
        exact_tokens = prompt_tokens + samples[sampled_name]["tokens"]
        exact_data = [_ce_datum_from_tokens(exact_tokens)]
        exact_results, elapsed = _await_pair(
            f"sample_association_forward_{sampled_name}",
            {
                name: clients[name].forward(exact_data, "cross_entropy")
                for name in ADAPTER_RANKS
            },
            args.timeout_s,
        )
        association_forward[sampled_name] = {
            name: _check_forward_result(
                name, exact_results[name], len(exact_tokens) - 1
            )
            for name in ADAPTER_RANKS
        }
        association_forward_seconds[sampled_name] = elapsed
    progress["association_forward"] = association_forward
    _log(
        "association diagnostics: "
        + json.dumps({"samples": samples, "training_scores": association_forward})
    )
    published_association = _assert_published_association(samples, association_forward)
    progress["published_association"] = published_association

    phase_seconds["forward"] = round(forward_seconds, 3)
    phase_seconds["sample"] = round(sample_seconds, 3)
    phase_seconds.update(
        {
            f"sample_association_forward_{name}": round(elapsed, 3)
            for name, elapsed in association_forward_seconds.items()
        }
    )

    return {
        "ok": True,
        "scope": "mechanical official-SDK smoke only; no routing replay or RL-quality claim",
        **progress,
        "phase_seconds": phase_seconds,
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "glm52_full_dual_client_sdk_smoke_summary.json"
    failure_path = args.out_dir / "glm52_full_dual_client_sdk_smoke_failure.json"
    summary_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    installed_sdk = version("tinker")
    if installed_sdk != SDK_VERSION:
        raise RuntimeError(
            f"this gate requires official tinker=={SDK_VERSION}, found {installed_sdk}"
        )
    clients: dict[str, Any] = {}
    progress: dict[str, Any] = {}
    try:
        summary = _run_smoke(args, clients, progress)
    except BaseException as exc:
        failure = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "progress": progress,
        }
        if clients:
            try:
                _unload_model_ids(
                    args.base_url,
                    args.api_key,
                    [client.model_id for client in clients.values()],
                    args.timeout_s,
                )
            except (
                Exception
            ) as cleanup_error:  # noqa: BLE001 - retain the primary smoke failure
                _log(f"cleanup after smoke failure also failed: {cleanup_error}")
                failure["cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True, default=str) + "\n"
        )
        _log(f"failure diagnostics written to {failure_path}")
        raise
    _unload_model_ids(
        args.base_url,
        args.api_key,
        [client.model_id for client in clients.values()],
        args.timeout_s,
    )

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _log(f"summary written to {summary_path}")
    print(
        "GLM52_TINKER_DUAL_CLIENT_SDK_SMOKE_PASS="
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
