import json
import shlex

import examples.multi_lora_operations.run_glm52_full_multi_lora_operations as launcher
import pytest
import yaml
from examples.multi_lora_operations.run_glm52_full_multi_lora_operations import (
    ScriptArgs,
    _require_external_ray,
    _rollout_checkpoint,
    _runtime_env,
    _train_args,
    _validate_full_checkpoint,
    _validate_topology,
)


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def test_full_glm52_tinker_service_command_contract(monkeypatch) -> None:
    args = ScriptArgs(run_id="unit-test")
    _validate_topology(args)
    argv = shlex.split(_train_args(args))

    assert {
        "--tinker-backend",
        "--tinker-frontend",
        "--use-dynamic-batch-size",
        "--sequence-parallel",
        "--sglang-enable-dp-attention",
    } <= set(argv)
    assert {
        option: _option_value(argv, option)
        for option in (
            "--hf-checkpoint",
            "--actor-num-nodes",
            "--actor-num-gpus-per-node",
            "--rollout-num-gpus",
            "--rollout-num-gpus-per-engine",
            "--sglang-ep-size",
            "--sglang-dp-size",
            "--tensor-model-parallel-size",
            "--expert-model-parallel-size",
            "--pipeline-model-parallel-size",
            "--context-parallel-size",
            "--expert-tensor-parallel-size",
            "--qkv-format",
            "--global-batch-size",
            "--multi-lora-n-adapters",
            "--tinker-sampling-max-context",
            "--sglang-context-length",
            "--sglang-router-port",
            "--target-modules",
            "--sglang-config",
        )
    } == {
        "--hf-checkpoint": "/cluster-storage/models/GLM-5.2",
        "--actor-num-nodes": "4",
        "--actor-num-gpus-per-node": "8",
        "--rollout-num-gpus": "16",
        "--rollout-num-gpus-per-engine": "8",
        "--sglang-ep-size": "8",
        "--sglang-dp-size": "8",
        "--tensor-model-parallel-size": "8",
        "--expert-model-parallel-size": "32",
        "--pipeline-model-parallel-size": "1",
        "--context-parallel-size": "1",
        "--expert-tensor-parallel-size": "1",
        "--qkv-format": "thd",
        "--global-batch-size": "4",
        "--multi-lora-n-adapters": "2",
        "--tinker-sampling-max-context": "8192",
        "--sglang-context-length": "8192",
        "--sglang-router-port": "18080",
        "--target-modules": "q_proj,k_proj,v_proj,o_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj",
        "--sglang-config": args.sglang_config,
    }
    storage_rank = int(_option_value(argv, "--lora-rank"))
    assert storage_rank == 16
    assert int(_option_value(argv, "--sglang-max-lora-rank")) == storage_rank
    assert {
        "--debug-train-only",
        "--colocate",
        "--multi-lora-adapter",
        "--multi-lora-disable-service-mode",
        "--use-rollout-routing-replay",
    }.isdisjoint(argv)
    assert _rollout_checkpoint(args) == "/cluster-storage/models/GLM-5.2_fp8"

    monkeypatch.setenv("MILES_TINKER_API_KEY", "unit-test-key")
    assert _runtime_env()["MILES_TINKER_API_KEY"] == "unit-test-key"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param({"actor_num_nodes": 3}, "four 8xH200 trainer nodes", id="trainer-nodes"),
        pytest.param({"num_gpus_per_node": 4}, "four 8xH200 trainer nodes", id="trainer-gpus"),
        pytest.param(
            {"rollout_num_gpus": 8},
            r"two 8xH200 engines \(16 GPUs total\)",
            id="rollout-total",
        ),
        pytest.param(
            {"rollout_num_gpus_per_engine": 4},
            r"two 8xH200 engines \(16 GPUs total\)",
            id="rollout-engine",
        ),
        pytest.param({"n_adapters": 1}, "at least two adapter slots", id="adapter-slots"),
        pytest.param({"lora_rank": 8}, "pins the storage and serving rank to 16", id="storage-rank-low"),
        pytest.param({"lora_rank": 24}, "pins the storage and serving rank to 16", id="storage-rank-high"),
    ),
)
def test_full_glm52_topology_validation(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_topology(ScriptArgs(**overrides))


def test_full_glm52_requires_external_ray(monkeypatch) -> None:
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    with pytest.raises(RuntimeError, match="existing 48-GPU Ray cluster"):
        _require_external_ray()

    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    five_nodes = [{"Alive": True, "Resources": {"GPU": 8, "accelerator_type:H200": 1}} for _ in range(5)]
    monkeypatch.setattr(launcher, "_live_ray_nodes", lambda: five_nodes)
    with pytest.raises(RuntimeError, match="found 5"):
        _require_external_ray()

    six_nodes = five_nodes + [{"Alive": True, "Resources": {"GPU": 8, "accelerator_type:H200": 1}}]
    monkeypatch.setattr(launcher, "_live_ray_nodes", lambda: six_nodes)
    _require_external_ray()

    monkeypatch.setattr(
        launcher,
        "_live_ray_nodes",
        lambda: six_nodes + [{"Alive": True, "Resources": {"GPU": 8, "accelerator_type:H200": 1}}],
    )
    with pytest.raises(RuntimeError, match="found 7"):
        _require_external_ray()

    mixed_nodes = five_nodes + [
        {"Alive": True, "Resources": {"GPU": 8, "accelerator_type:A100": 1}},
    ]
    monkeypatch.setattr(launcher, "_live_ray_nodes", lambda: mixed_nodes)
    with pytest.raises(RuntimeError, match="every live GPU node"):
        _require_external_ray()


def test_full_glm52_rollout_config_rejects_four_gpu_engines(tmp_path) -> None:
    args = ScriptArgs()
    with open(args.sglang_config) as config_file:
        config = yaml.safe_load(config_file)
    config["sglang"][0]["num_gpus_per_engine"] = 4
    config_path = tmp_path / "split-engines.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(
        ValueError,
        match="one regular rollout server group totaling 16 GPUs with 8 GPUs per engine",
    ):
        _rollout_checkpoint(ScriptArgs(sglang_config=str(config_path)))


def test_full_glm52_rollout_config_rejects_one_engine_total(tmp_path) -> None:
    args = ScriptArgs()
    with open(args.sglang_config) as config_file:
        config = yaml.safe_load(config_file)
    config["sglang"][0]["server_groups"][0]["num_gpus"] = 8
    config_path = tmp_path / "one-engine.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(
        ValueError,
        match="one regular rollout server group totaling 16 GPUs with 8 GPUs per engine",
    ):
        _rollout_checkpoint(ScriptArgs(sglang_config=str(config_path)))


@pytest.mark.parametrize(
    ("directory", "updates", "message"),
    (
        pytest.param("GLM-5.2", {}, None, id="native-full"),
        pytest.param("GLM-5.2_5layer", {}, "reduced GLM-5.2", id="reduced-name"),
        pytest.param("GLM-5.2", {"num_hidden_layers": 5}, "78 layers", id="pruned-layers"),
        pytest.param(
            "GLM-5.2",
            {"num_experts_per_tok": 4},
            "top-8 of 256 routed experts",
            id="routing-shape",
        ),
        pytest.param(
            "GLM-5.2",
            {"auto_map": {"AutoModel": "modeling_glm.GlmModel"}},
            "auto_map",
            id="custom-model-code",
        ),
    ),
)
def test_full_glm52_checkpoint_validation(tmp_path, directory: str, updates: dict, message: str | None) -> None:
    checkpoint = tmp_path / directory
    checkpoint.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 78,
        "num_experts_per_tok": 8,
        "n_routed_experts": 256,
    }
    config.update(updates)
    (checkpoint / "config.json").write_text(json.dumps(config))

    if message is None:
        _validate_full_checkpoint(str(checkpoint))
    else:
        with pytest.raises(ValueError, match=message):
            _validate_full_checkpoint(str(checkpoint))


def test_full_glm52_checkpoint_precision_roles(tmp_path) -> None:
    checkpoint = tmp_path / "GLM-5.2"
    checkpoint.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 78,
        "num_experts_per_tok": 8,
        "n_routed_experts": 256,
    }
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps(config))

    _validate_full_checkpoint(str(checkpoint))
    with pytest.raises(ValueError, match="rollout checkpoint must declare"):
        _validate_full_checkpoint(str(checkpoint), require_fp8=True)

    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3"}
    config_path.write_text(json.dumps(config))
    _validate_full_checkpoint(str(checkpoint), require_fp8=True)
    with pytest.raises(ValueError, match="trainer checkpoint must be unquantized"):
        _validate_full_checkpoint(str(checkpoint))
