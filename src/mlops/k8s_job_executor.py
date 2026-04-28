from __future__ import annotations


DEFAULT_NAMESPACE = "admin"
VALID_RESTART_POLICIES = {"Never", "OnFailure"}


def validate_k8s_job_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Kubernetes Job manifest must be a dictionary.")
    if manifest.get("apiVersion") != "batch/v1":
        raise ValueError("Kubernetes Job manifest apiVersion must be 'batch/v1'.")
    if manifest.get("kind") != "Job":
        raise ValueError("Kubernetes Job manifest kind must be 'Job'.")

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Kubernetes Job manifest metadata must be a dictionary.")
    if not metadata.get("name"):
        raise ValueError("Kubernetes Job manifest metadata.name is required.")
    metadata.setdefault("namespace", DEFAULT_NAMESPACE)

    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("Kubernetes Job manifest spec is required.")

    template = spec.get("template")
    if not isinstance(template, dict):
        raise ValueError("Kubernetes Job manifest spec.template is required.")

    pod_spec = template.get("spec")
    if not isinstance(pod_spec, dict):
        raise ValueError("Kubernetes Job manifest spec.template.spec is required.")

    containers = pod_spec.get("containers")
    if not isinstance(containers, list) or not containers:
        raise ValueError(
            "Kubernetes Job manifest spec.template.spec.containers must be a non-empty list."
        )

    restart_policy = pod_spec.get("restartPolicy")
    if restart_policy not in VALID_RESTART_POLICIES:
        raise ValueError(
            "Kubernetes Job manifest spec.template.spec.restartPolicy must be 'Never' or 'OnFailure'."
        )


def get_k8s_job_namespace(manifest: dict) -> str:
    metadata = manifest.get("metadata") or {}
    return metadata.get("namespace") or DEFAULT_NAMESPACE


def get_k8s_job_name(manifest: dict) -> str:
    metadata = manifest.get("metadata") or {}
    return metadata["name"]


def create_k8s_job(manifest: dict):
    """Create a Kubernetes Job from a manifest dict using kubeconfig auth."""
    validate_k8s_job_manifest(manifest)
    namespace = get_k8s_job_namespace(manifest)

    from kubernetes import client, config

    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    return batch_v1.create_namespaced_job(
        namespace=namespace,
        body=manifest,
    )
