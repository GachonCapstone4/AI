from __future__ import annotations

import argparse
import json
import sys

import pytest

from launcher import run as launcher_run
from src.mlops import k8s_job_executor


def _manifest(**overrides: object) -> dict:
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "sample", "namespace": "admin"},
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "diag-tool",
                            "image": "suhannugul/capstonetest:latest",
                            "args": ["network"],
                        }
                    ],
                }
            }
        },
    }
    manifest.update(overrides)
    return manifest


def test_load_k8s_job_manifest_from_json() -> None:
    args = argparse.Namespace(
        manifest_json=json.dumps(_manifest()),
        manifest_path=None,
    )

    manifest = launcher_run.load_k8s_job_manifest(args)

    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "sample"


def test_load_k8s_job_manifest_from_yaml(tmp_path) -> None:
    manifest_path = tmp_path / "job.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "apiVersion: batch/v1",
                "kind: Job",
                "metadata:",
                "  name: sample-yaml",
                "spec:",
                "  template:",
                "    spec:",
                "      restartPolicy: Never",
                "      containers:",
                "        - name: diag-tool",
                "          image: suhannugul/capstonetest:latest",
                "          args: [\"network\"]",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(manifest_json=None, manifest_path=str(manifest_path))

    manifest = launcher_run.load_k8s_job_manifest(args)

    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "sample-yaml"


def test_load_k8s_job_manifest_requires_single_source() -> None:
    args = argparse.Namespace(manifest_json="{}", manifest_path="job.yaml")

    with pytest.raises(ValueError, match="Use only one"):
        launcher_run.load_k8s_job_manifest(args)


def test_load_k8s_job_manifest_requires_source() -> None:
    args = argparse.Namespace(manifest_json=None, manifest_path=None)

    with pytest.raises(ValueError, match="requires --manifest-json or --manifest-path"):
        launcher_run.load_k8s_job_manifest(args)


def test_validate_k8s_job_manifest_rejects_non_job_kind() -> None:
    with pytest.raises(ValueError, match="kind must be 'Job'"):
        k8s_job_executor.validate_k8s_job_manifest(_manifest(kind="Deployment"))


def test_validate_k8s_job_manifest_rejects_wrong_api_version() -> None:
    with pytest.raises(ValueError, match="apiVersion must be 'batch/v1'"):
        k8s_job_executor.validate_k8s_job_manifest(_manifest(apiVersion="apps/v1"))


def test_validate_k8s_job_manifest_requires_metadata_name() -> None:
    with pytest.raises(ValueError, match="metadata.name is required"):
        k8s_job_executor.validate_k8s_job_manifest(_manifest(metadata={}))


def test_validate_k8s_job_manifest_requires_spec() -> None:
    manifest = _manifest()
    del manifest["spec"]

    with pytest.raises(ValueError, match="spec is required"):
        k8s_job_executor.validate_k8s_job_manifest(manifest)


def test_validate_k8s_job_manifest_requires_non_empty_containers() -> None:
    manifest = _manifest(
        spec={"template": {"spec": {"restartPolicy": "Never", "containers": []}}}
    )

    with pytest.raises(ValueError, match="containers must be a non-empty list"):
        k8s_job_executor.validate_k8s_job_manifest(manifest)


def test_validate_k8s_job_manifest_requires_restart_policy() -> None:
    manifest = _manifest(
        spec={
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "diag-tool",
                            "image": "suhannugul/capstonetest:latest",
                            "args": ["network"],
                        }
                    ]
                }
            }
        }
    )

    with pytest.raises(ValueError, match="restartPolicy must be 'Never' or 'OnFailure'"):
        k8s_job_executor.validate_k8s_job_manifest(manifest)


def test_validate_k8s_job_manifest_rejects_invalid_restart_policy() -> None:
    manifest = _manifest(
        spec={
            "template": {
                "spec": {
                    "restartPolicy": "Always",
                    "containers": [{"name": "diag-tool"}],
                }
            }
        }
    )

    with pytest.raises(ValueError, match="restartPolicy must be 'Never' or 'OnFailure'"):
        k8s_job_executor.validate_k8s_job_manifest(manifest)


def test_namespace_defaults_to_admin() -> None:
    manifest = _manifest(metadata={"name": "sample"})

    k8s_job_executor.validate_k8s_job_manifest(manifest)

    assert k8s_job_executor.get_k8s_job_namespace(manifest) == "admin"
    assert manifest["metadata"]["namespace"] == "admin"


def test_namespace_default_does_not_override_existing_namespace() -> None:
    manifest = _manifest(metadata={"name": "sample", "namespace": "ops"})

    k8s_job_executor.validate_k8s_job_manifest(manifest)

    assert manifest["metadata"]["namespace"] == "ops"


def test_create_k8s_job_uses_namespace_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeConfig:
        @staticmethod
        def load_kube_config() -> None:
            calls["loaded"] = True

    class FakeBatchV1Api:
        def create_namespaced_job(self, namespace: str, body: dict) -> object:
            calls["namespace"] = namespace
            calls["body"] = body
            return object()

    class FakeClient:
        BatchV1Api = FakeBatchV1Api

    monkeypatch.setitem(
        sys.modules,
        "kubernetes",
        type("FakeKubernetes", (), {"client": FakeClient, "config": FakeConfig}),
    )

    manifest = _manifest()
    k8s_job_executor.create_k8s_job(manifest)

    assert calls == {"loaded": True, "namespace": "admin", "body": manifest}


def test_k8s_dry_run_output_contains_job_metadata() -> None:
    output = launcher_run.build_k8s_job_dry_run_output(_manifest())

    assert output["job_type"] == "k8s_job"
    assert output["dry_run"] is True
    assert output["k8s_job_name"] == "sample"
    assert output["namespace"] == "admin"
    assert output["manifest"]["kind"] == "Job"


def test_k8s_dry_run_output_reflects_default_namespace_in_manifest() -> None:
    manifest = _manifest(metadata={"name": "sample"})

    output = launcher_run.build_k8s_job_dry_run_output(manifest)

    assert output["namespace"] == "admin"
    assert output["manifest"]["metadata"]["namespace"] == "admin"


def test_training_dry_run_does_not_require_manifest(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--job-id", "training-001", "--job-type", "training", "--dry-run"],
    )

    launcher_run.main()
    output = json.loads(capsys.readouterr().out)

    assert output["job_type"] == "training"
    assert output["dry_run"] is True
    assert "training_job_config" in output
