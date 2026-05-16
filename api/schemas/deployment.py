from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DeploymentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(validation_alias=AliasChoices("job_id", "jobId"))
    model_version: str = Field(validation_alias=AliasChoices("model_version", "modelVersion"))
    artifact_s3_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("artifact_s3_uri", "artifactS3Uri"),
    )
    requested_by: str | None = Field(
        default=None,
        validation_alias=AliasChoices("requested_by", "requestedBy"),
    )
    requested_at: str | None = Field(
        default=None,
        validation_alias=AliasChoices("requested_at", "requestedAt"),
    )


class DeploymentRunningEvent(BaseModel):
    job_id: str
    status: Literal["RUNNING"]
    model_version: str
    stage: Literal["PRELOAD", "VALIDATE", "SWITCH"]
    error_message: str | None = None
    finished_at: str | None = None
    message: str
    timestamp: str


class DeploymentCompletedEvent(BaseModel):
    job_id: str
    status: Literal["COMPLETED"]
    model_version: str
    active_model_version: str
    stage: Literal["COMPLETED"] = "COMPLETED"
    error_message: str | None = None
    finished_at: str
    message: str = "Deployment completed"


class DeploymentFailedEvent(BaseModel):
    job_id: str
    status: Literal["FAILED"]
    model_version: str
    stage: str
    error_message: str
    finished_at: str
