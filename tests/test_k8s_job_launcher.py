from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

from launcher import run as launcher_run
from src.mlops import k8s_job_executor


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_k8s_dry_run_injects_launcher_job_id(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    manifest = _manifest()
    manifest["spec"]["template"]["spec"]["containers"][0]["env"] = [
        {"name": "ADMIN_USER_ID", "value": "54"}
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--job-id",
            "dataset-batch-from-launcher",
            "--job-type",
            "k8s_job",
            "--dry-run",
            "--manifest-json",
            json.dumps(manifest),
        ],
    )

    launcher_run.main()
    output = json.loads(capsys.readouterr().out)
    env = output["manifest"]["spec"]["template"]["spec"]["containers"][0]["env"]

    assert {"name": "JOB_ID", "value": "dataset-batch-from-launcher"} in env
    assert output["k8s_job_name"] == "dataset-batch-from-launcher"


def test_k8s_dry_run_generates_job_id_when_missing(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--job-type",
            "k8s_job",
            "--dry-run",
            "--manifest-json",
            json.dumps(manifest),
        ],
    )

    launcher_run.main()
    output = json.loads(capsys.readouterr().out)
    env = output["manifest"]["spec"]["template"]["spec"]["containers"][0]["env"]
    job_id = next(item["value"] for item in env if item["name"] == "JOB_ID")

    assert re.fullmatch(r"dataset-batch-\d{14}", job_id)
    assert output["k8s_job_name"] == job_id
    assert output["manifest"]["metadata"]["name"] == job_id


def test_training_requires_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--job-type", "training", "--dry-run"],
    )

    with pytest.raises(SystemExit) as exc_info:
        launcher_run.parse_args()

    assert exc_info.value.code == 2


def test_k8s_job_id_injection_overwrites_existing_value() -> None:
    manifest = _manifest()
    manifest["spec"]["template"]["spec"]["containers"][0]["env"] = [
        {"name": "JOB_ID", "value": "stale-job-id"},
        {"name": "ADMIN_USER_ID", "value": "54"},
    ]

    launcher_run.inject_k8s_job_id(manifest, "fresh-job-id")

    env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    job_id_entries = [item for item in env if item["name"] == "JOB_ID"]
    assert job_id_entries == [{"name": "JOB_ID", "value": "fresh-job-id"}]


def test_dataset_batch_manifest_uses_dataset_batch_image_tag() -> None:
    yaml = pytest.importorskip("yaml")
    manifest_path = REPO_ROOT / "manifests" / "dataset-batch.yaml"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    image = manifest["spec"]["template"]["spec"]["containers"][0]["image"]
    assert image == (
        "390403881443.dkr.ecr.ap-northeast-2.amazonaws.com/capstone/ecr:dataset-batch"
    )


def test_dataset_batch_manifest_uses_admin_user_id_1() -> None:
    yaml = pytest.importorskip("yaml")
    manifest_path = REPO_ROOT / "manifests" / "dataset-batch.yaml"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]

    assert {"name": "ADMIN_USER_ID", "value": "1"} in env


def test_dataset_batch_training_event_uses_uppercase_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mysql", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mysql.connector", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pika", types.SimpleNamespace())
    dataset_batch = importlib.import_module("batch.dataset_batch")

    monkeypatch.setattr(dataset_batch, "JOB_ID", "dataset-batch-test")
    published = []

    class FakeProperties:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePika:
        BasicProperties = FakeProperties

    class FakeChannel:
        def queue_declare(self, **kwargs):
            pass

        def basic_publish(self, **kwargs):
            published.append(kwargs)

    monkeypatch.setattr(dataset_batch, "pika", FakePika)

    dataset_batch.publish_training_event(FakeChannel(), "COMPLETED", dataset_version="v1")
    dataset_batch.publish_training_event(FakeChannel(), "FAILED", error_message="boom")

    payloads = [json.loads(item["body"]) for item in published]
    assert payloads[0]["status"] == "COMPLETED"
    assert payloads[1]["status"] == "FAILED"


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
