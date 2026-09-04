"""Static distribution and donor-provenance contracts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APACHE_2_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
OMLX_SHA = "e467261edc786efd33b1e9023d5c4a827f8aa1c1"
MTPLX_SHA = "6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab"
REQUIRED_DISTRIBUTION_FILES = {
    "LICENSE",
    "NOTICE",
    "LICENSES/Apache-2.0.txt",
}
EXPECTED_DERIVED_PATHS = {
    "responses/registry.py",
    "runtime/events.py",
    "runtime/fusion/cache/contracts.py",
    "runtime/fusion/cache/identity.py",
    "runtime/fusion/cache/lifecycle.py",
    "runtime/fusion/concrete/owner.py",
    "runtime/fusion/concrete/stepper.py",
    "runtime/fusion/mtp/contracts.py",
    "runtime/fusion/mtp/policy.py",
    "runtime/fusion/qwen4_exp/cache_adapter.py",
    "runtime/fusion/qwen4_exp/execution.py",
    "runtime/fusion/qwen4_exp/model/config.py",
    "runtime/fusion/qwen4_exp/model/qsa_replay.py",
    "runtime/fusion/qwen4_exp/model/sampling.py",
    "runtime/fusion/qwen4_exp/model/tensor.py",
    "runtime/fusion/qwen4_exp/model/tensor_support.py",
    "runtime/fusion/qwen4_exp/prefix_store.py",
    "runtime/fusion/qwen4_exp/probe.py",
    "runtime/fusion/qwen4_exp/vision/mrope.py",
    "runtime/fusion/qwen4_exp/vision/processing.py",
    "runtime/fusion/qwen4_exp/vision/splice.py",
    "runtime/fusion/qwen4_exp/vision/tensor_mrope.py",
    "runtime/fusion/qwen4_exp/vision/tensor_processing.py",
    "runtime/fusion/qwen4_exp/vision/tensor_splice.py",
    "runtime/fusion/qwen4_exp/vision/tensor_tower.py",
    "runtime/fusion/qwen4_exp/vision/tower.py",
    "runtime/fusion/scheduler/chassis.py",
    "runtime/fusion/scheduler/contracts.py",
    "runtime/turn.py",
    "tools/dialects/harmony.py",
    "tools/dialects/qwen.py",
}


def test_wheel_and_sdist_declare_required_license_files() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert set(metadata["project"]["license-files"]) >= REQUIRED_DISTRIBUTION_FILES
    sdist = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert set(sdist["include"]) >= REQUIRED_DISTRIBUTION_FILES

    wheel = metadata["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "src/mlx_batch_server" in wheel["include"]
    assert (ROOT / "src/mlx_batch_server/SOURCE_ORIGINS.json").is_file()
    assert (
        ROOT / "src/mlx_batch_server/runtime/manifests/runtime-roles-8100-8102.json"
    ).is_file()


def test_bundled_apache_license_matches_frozen_mtplx_license() -> None:
    bundled_license = (ROOT / "LICENSES/Apache-2.0.txt").read_bytes()

    assert hashlib.sha256(bundled_license).hexdigest() == APACHE_2_SHA256


def test_origin_manifest_is_complete_and_uses_frozen_donor_revisions() -> None:
    manifest = json.loads(
        (ROOT / "src/mlx_batch_server/SOURCE_ORIGINS.json").read_text(encoding="utf-8")
    )

    assert manifest["donors"]["omlx"]["commit"] == OMLX_SHA
    assert manifest["donors"]["mtplx"]["commit"] == MTPLX_SHA
    assert {
        item["project"] for item in manifest["donors"]["omlx"]["upstream_attributions"]
    } == {"vllm-mlx", "vLLM v1"}
    assert manifest["distribution_files"] == {
        "project_license": "LICENSE",
        "third_party_license": "LICENSES/Apache-2.0.txt",
        "notices": "NOTICE",
    }

    derived_files = manifest["derived_files"]
    assert {record["target_path"] for record in derived_files} == EXPECTED_DERIVED_PATHS

    for record in derived_files:
        assert record["source_commit"] in {OMLX_SHA, MTPLX_SHA}
        assert record["license"] == "Apache-2.0"
        assert (ROOT / "src/mlx_batch_server" / record["target_path"]).is_file()


def test_notice_retains_mtplx_and_vllm_attribution() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "Powered by MTPLX" in notice
    assert "https://github.com/youssofal/mtplx" in notice
    assert "Copyright 2025 oMLX contributors" in notice
    assert "vllm-mlx" in notice
    assert "vLLM v1" in notice
