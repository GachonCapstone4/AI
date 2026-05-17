# ============================================================
# 메시지 계약 테스트 (Message Contract Tests)
#
# RabbitMQ / FastAPI 없이 순수 스키마 레벨 검증
#
# 커버 범위
# ----------
# q.2ai.classify  입력 파싱  (ClassifyRequest)
# q.2app.classify 출력 검증 (ClassifyResponse + ResponseMeta)
# ============================================================

import json
import pytest
from pydantic import ValidationError

from api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    Classification,
    DeploymentCompletedEvent,
    DeploymentFailedEvent,
    DeploymentRequest,
    DeploymentRunningEvent,
    ResponseMeta,
)
from messaging.consumer_classify import _build_backend_classify_payload
from messaging.consumer_deployment import (
    CONSUME_QUEUE,
    SOURCE_EXCHANGE,
    SOURCE_ROUTING_KEY,
    PUBLISH_EXCHANGE,
    PUBLISH_QUEUE,
    PUBLISH_ROUTING_KEY,
    CONSUME_QUEUE_ARGUMENTS,
    SSE_EXCHANGE,
    SSE_FAILURE_MESSAGE,
    SSE_SUCCESS_MESSAGE,
    DeploymentConsumerRunner,
    declare_deployment_topology,
    normalize_deployment_payload,
    process_deployment_message,
)
from messaging.publisher import (
    AI2APP_EXCHANGE,
    DEPLOYMENT_STATUS_QUEUE,
    JOB_STATUS_ROUTING_KEY,
    publish,
)


# ── 픽스처: 표준 입력 메시지 ─────────────────────────────────

@pytest.fixture
def classify_input():
    return {
        "outbox_id":    1,
        "email_id":     1,
        "sender_email": "sender@example.com",
        "sender_name":  "홍길동",
        "subject":      "납품 일정 문의",
        "body_clean":   "이번 달 납품 일정을 알려주시겠어요?",
        "received_at":  "2026-04-06T10:00:00",
    }


@pytest.fixture
def classify_output():
    return {
        "outbox_id":       1,
        "email_id":        1,
        "classification":  {"domain": "업무", "intent": "문의"},
        "confidence_score": 0.91,
        "summary":         "납품 일정 확인 요청 이메일입니다.",
        "schedule_info":   None,
        "email_embedding": [0.1, 0.2, 0.3],
    }

# ── q.2ai.classify 입력 파싱 ─────────────────────────────────

class TestClassifyInput:
    def test_valid_message_parses(self, classify_input):
        req = ClassifyRequest(**classify_input)
        assert req.outbox_id    == 1
        assert req.email_id     == 1
        assert req.subject      == "납품 일정 문의"
        assert req.body_clean   == "이번 달 납품 일정을 알려주시겠어요?"

    def test_missing_outbox_id_raises(self, classify_input):
        classify_input.pop("outbox_id")
        with pytest.raises(ValidationError):
            ClassifyRequest(**classify_input)

    def test_missing_subject_raises(self, classify_input):
        classify_input.pop("subject")
        with pytest.raises(ValidationError):
            ClassifyRequest(**classify_input)

    def test_missing_body_clean_raises(self, classify_input):
        classify_input.pop("body_clean")
        with pytest.raises(ValidationError):
            ClassifyRequest(**classify_input)

    def test_json_roundtrip(self, classify_input):
        """JSON 직렬화 → 역직렬화 무결성"""
        raw  = json.dumps(classify_input)
        data = json.loads(raw)
        req  = ClassifyRequest(**data)
        assert req.email_id == classify_input["email_id"]

    def test_camel_case_aliases_parse(self):
        req = ClassifyRequest(
            outboxId=7,
            emailId=11,
            senderEmail="sender@example.com",
            senderName="홍길동",
            subject="회의 일정 안내",
            bodyClean="정제된 본문...",
            receivedAt="2026-04-06T10:00:00",
        )
        assert req.outbox_id == 7
        assert req.email_id == 11


# ── q.2app.classify 출력 검증 ────────────────────────────────

class TestClassifyOutput:
    def test_valid_response_parses(self, classify_output):
        resp = ClassifyResponse(**classify_output)
        assert resp.outbox_id == 1
        assert resp.classification.domain == "업무"
        assert resp.classification.intent == "문의"
        assert resp.confidence_score == 0.91
        assert isinstance(resp.email_embedding, list)
        assert all(isinstance(v, float) for v in resp.email_embedding)

    def test_schedule_info_optional(self, classify_output):
        classify_output["schedule_info"] = None
        resp = ClassifyResponse(**classify_output)
        assert resp.schedule_info is None

    def test_schedule_info_with_dict(self, classify_output):
        classify_output["schedule_info"] = {
            "date": "2026-04-10", "time": "14:00",
            "location": "회의실 A",
        }
        resp = ClassifyResponse(**classify_output)
        assert resp.schedule_info["date"] == "2026-04-10"
        assert "attendees" not in resp.schedule_info

    def test_embedding_must_be_float_list(self, classify_output):
        classify_output["email_embedding"] = [0.1, 0.2, 0.3]
        resp = ClassifyResponse(**classify_output)
        assert len(resp.email_embedding) == 3

    def test_outbox_id_preserved(self, classify_output):
        """outbox_id 가 입력과 동일하게 출력에 포함되어야 함"""
        resp = ClassifyResponse(**classify_output)
        assert resp.outbox_id == classify_output["outbox_id"]

    def test_json_serializable(self, classify_output):
        resp = ClassifyResponse(**classify_output)
        dumped = resp.model_dump()
        raw = json.dumps(dumped)            # JSON 직렬화 가능해야 함
        assert "outbox_id" in json.loads(raw)

# ── ResponseMeta 검증 ────────────────────────────────────────

class TestResponseMeta:
    def test_valid_meta_parses(self):
        meta = ResponseMeta(
            elapsed_ms=123.4,
            source="consumer.classify",
            model_version="2026-04-14-001",
        )
        assert meta.elapsed_ms == 123.4
        assert meta.source     == "consumer.classify"
        assert meta.model_version == "2026-04-14-001"

    def test_source_defaults_to_ai_server(self):
        meta = ResponseMeta(elapsed_ms=50.0)
        assert meta.source == "ai-server"

    def test_meta_embedded_in_classify_response(self, classify_output):
        classify_output["meta"] = {
            "elapsed_ms": 99.9,
            "source": "consumer.classify",
            "model_version": "2026-04-14-001",
        }
        resp = ClassifyResponse(**classify_output)
        assert resp.meta is not None
        assert resp.meta.elapsed_ms == 99.9
        assert resp.meta.source     == "consumer.classify"
        assert resp.meta.model_version == "2026-04-14-001"

    def test_meta_absent_is_none(self, classify_output):
        resp = ClassifyResponse(**classify_output)
        assert resp.meta is None

    def test_meta_json_serializable(self, classify_output):
        classify_output["meta"] = {
            "elapsed_ms": 77.0,
            "source": "consumer.classify",
            "model_version": "2026-04-14-001",
        }
        resp = ClassifyResponse(**classify_output)
        dumped = json.dumps(resp.model_dump())
        parsed = json.loads(dumped)
        assert parsed["meta"]["elapsed_ms"] == 77.0
        assert parsed["meta"]["model_version"] == "2026-04-14-001"


class TestBackendClassifyPublishPayload:
    def test_builds_flat_backend_contract_with_defaults(self):
        result = ClassifyResponse(
            outbox_id=1,
            email_id=2,
            classification=Classification(domain="업무", intent="문의"),
            confidence_score=0.91,
            summary="납품 일정 확인 요청 이메일입니다.",
            schedule_info={
                "date": "2026-04-10",
                "time": "14:00",
                "location": "회의실 A",
                "attendees": ["kim@example.com"],
            },
            email_embedding=[0.1, 0.2, 0.3],
            meta=ResponseMeta(
                elapsed_ms=12.3,
                source="consumer.classify",
                model_version="2026-04-14-001",
            ),
        )

        payload = _build_backend_classify_payload(result)

        assert payload == {
            "outbox_id": 1,
            "email_id": 2,
            "domain": "업무",
            "intent": "문의",
            "confidence_score": 0.91,
            "summary_text": "납품 일정 확인 요청 이메일입니다.",
            "schedule_detected": True,
            "entities_json": {
                "date": "2026-04-10",
                "time": "14:00",
                "location": "회의실 A",
            },
            "model_version": "2026-04-14-001",
        }
        assert isinstance(payload["entities_json"], dict)
        assert set(payload["entities_json"].keys()) == {"date", "time", "location"}
        assert "attendees" not in payload["entities_json"]

    def test_builds_empty_schedule_defaults(self):
        result = ClassifyResponse(
            outbox_id=1,
            email_id=2,
            classification=Classification(domain="업무", intent="문의"),
            confidence_score=0.45,
            summary="요약",
            schedule_info=None,
            email_embedding=[0.1],
        )

        payload = _build_backend_classify_payload(result)

        assert payload["schedule_detected"] is False
        assert payload["entities_json"] == {}

    def test_builds_empty_entities_when_only_location_exists(self):
        result = ClassifyResponse(
            outbox_id=1,
            email_id=2,
            classification=Classification(domain="업무", intent="문의"),
            confidence_score=0.45,
            summary="요약",
            schedule_info={"location": "Zoom", "attendees": ["kim@example.com"]},
            email_embedding=[0.1],
        )

        payload = _build_backend_classify_payload(result)

        assert payload["schedule_detected"] is False
        assert payload["entities_json"] == {}

    def test_publish_serializes_entities_json_as_json_object(self):
        class FakeChannel:
            def __init__(self):
                self.published = None

            def basic_publish(self, **kwargs):
                self.published = kwargs

        payload = {
            "outbox_id": 15,
            "email_id": 19,
            "domain": "Sales",
            "intent": "Meeting Request",
            "confidence_score": 0.84,
            "summary_text": "이메일 핵심 요약",
            "schedule_detected": True,
            "entities_json": {
                "date": "2026-04-21",
                "time": "14:00",
                "location": "Zoom",
            },
            "model_version": "2026-04-14-001",
        }

        channel = FakeChannel()
        publish(channel, "2app.classify", payload)

        published_body = json.loads(channel.published["body"].decode("utf-8"))
        assert isinstance(published_body["entities_json"], dict)
        assert published_body["entities_json"] == {
            "date": "2026-04-21",
            "time": "14:00",
            "location": "Zoom",
        }
        assert "attendees" not in published_body["entities_json"]


class TestDeploymentMessageContracts:
    def test_deployment_topology_constants_match_infra_binding(self):
        assert CONSUME_QUEUE == "q.2ai.deployment"
        assert SOURCE_EXCHANGE == "x.app2ai.direct"
        assert SOURCE_ROUTING_KEY == "2ai.deployment"
        assert CONSUME_QUEUE_ARGUMENTS == {
            "x-dead-letter-exchange": "x.retry.direct",
            "x-dead-letter-routing-key": "2ai.deployment.retry",
        }
        assert PUBLISH_EXCHANGE == "x.ai2app.direct"
        assert PUBLISH_QUEUE == "q.2app.training"
        assert PUBLISH_ROUTING_KEY == "app.training"
        assert DEPLOYMENT_STATUS_QUEUE == "q.2app.training"
        assert JOB_STATUS_ROUTING_KEY == "app.training"

    def test_deployment_consumer_declares_required_topology(self):
        class FakeChannel:
            def __init__(self):
                self.exchanges = []
                self.queues = []
                self.bindings = []

            def exchange_declare(self, **kwargs):
                self.exchanges.append(kwargs)

            def queue_declare(self, **kwargs):
                self.queues.append(kwargs)

            def queue_bind(self, **kwargs):
                self.bindings.append(kwargs)

        channel = FakeChannel()
        declare_deployment_topology(channel)

        assert {
            "exchange": "x.app2ai.direct",
            "exchange_type": "direct",
            "durable": True,
        } in channel.exchanges
        assert {
            "exchange": "x.ai2app.direct",
            "exchange_type": "direct",
            "durable": True,
        } in channel.exchanges
        assert {
            "queue": "q.2ai.deployment",
            "durable": True,
            "arguments": {
                "x-dead-letter-exchange": "x.retry.direct",
                "x-dead-letter-routing-key": "2ai.deployment.retry",
            },
        } in channel.queues
        assert {"queue": "q.2app.training", "durable": True} in channel.queues
        assert {
            "queue": "q.2ai.deployment",
            "exchange": "x.app2ai.direct",
            "routing_key": "2ai.deployment",
        } in channel.bindings
        assert {
            "queue": "q.2app.training",
            "exchange": "x.ai2app.direct",
            "routing_key": "app.training",
        } in channel.bindings

    def test_deployment_request_requires_job_id_and_model_version(self):
        payload = DeploymentRequest(
            job_id="deploy-2026-05-04-001",
            model_version="training-final-004",
            artifact_s3_uri="s3://capstone-gachon/models/training-final-004/",
            requested_by="admin",
            requested_at="2026-05-04T10:30:00Z",
        )

        assert payload.job_id == "deploy-2026-05-04-001"
        assert payload.model_version == "training-final-004"
        admin_payload = DeploymentRequest(user_id=1, task_type="deployment")
        assert admin_payload.user_id == 1
        assert admin_payload.task_type == "deployment"
        assert admin_payload.job_type is None
        assert admin_payload.job_id is None
        assert admin_payload.model_version is None

    def test_deployment_request_accepts_camel_case_aliases(self):
        payload = DeploymentRequest(
            jobId="deploy-1",
            modelVersion="training-final-004",
            artifactS3Uri="s3://bucket/models/training-final-004/",
            requestedBy="admin",
            requestedAt="2026-05-04T10:30:00Z",
        )

        assert payload.job_id == "deploy-1"
        assert payload.model_version == "training-final-004"
        assert payload.artifact_s3_uri == "s3://bucket/models/training-final-004/"

    def test_deployment_request_generates_fallback_job_id_for_admin_payload(self):
        payload = DeploymentRequest(user_id=1, task_type="deployment")

        normalized = normalize_deployment_payload(payload)

        assert normalized.job_id.startswith("deployment-1-")
        assert normalized.model_version is None

    def test_deployment_request_accepts_admin_model_job_payload(self):
        payload = DeploymentRequest(user_id=1, task_type="deployment", job_type="model")

        normalized = normalize_deployment_payload(payload)

        assert normalized.user_id == 1
        assert normalized.task_type == "deployment"
        assert normalized.job_type == "model"
        assert normalized.job_id.startswith("deployment-1-")
        assert normalized.model_version is None

    def test_deployment_request_defaults_missing_job_type_to_model_for_events(self):
        payload = DeploymentRequest(user_id=1, task_type="deployment")

        normalized = normalize_deployment_payload(payload)

        assert normalized.job_type == "model"

    def test_deployment_events_parse(self):
        running = DeploymentRunningEvent(
            job_id="deploy-1",
            job_type="model",
            status="RUNNING",
            model_version="training-final-004",
            stage="PRELOAD",
            message="Preloading deployment model",
            timestamp="2026-05-04T10:30:01Z",
        )
        completed = DeploymentCompletedEvent(
            job_id="deploy-1",
            job_type="model",
            status="COMPLETED",
            model_version="training-final-004",
            active_model_version="training-final-004",
            finished_at="2026-05-04T10:30:05Z",
        )
        failed = DeploymentFailedEvent(
            job_id="deploy-1",
            job_type="model",
            status="FAILED",
            model_version="training-final-004",
            stage="VALIDATE",
            error_message="validation failed",
            finished_at="2026-05-04T10:30:05Z",
        )

        assert running.stage == "PRELOAD"
        assert running.job_type == "model"
        assert completed.message == "Deployment completed"
        assert completed.stage == "COMPLETED"
        assert completed.error_message is None
        assert completed.job_type == "model"
        assert failed.status == "FAILED"
        assert failed.job_type == "model"

    def test_process_deployment_publishes_running_and_completed_events(self):
        class FakeChannel:
            def __init__(self):
                self.published = []

            def basic_publish(self, **kwargs):
                self.published.append(
                    {
                        "exchange": kwargs["exchange"],
                        "routing_key": kwargs["routing_key"],
                        "body": json.loads(kwargs["body"].decode("utf-8")),
                    }
                )

        class FakeManager:
            def __init__(self):
                self.calls = []

            def preload(self, version):
                self.calls.append(("preload", version))
                return {"status": "preloaded", "model_version": version}

            def validate(self):
                self.calls.append(("validate", None))
                return {"status": "validated", "model_version": "training-final-004"}

            def switch(self):
                self.calls.append(("switch", None))
                return {"status": "switched", "model_version": "training-final-004"}

        channel = FakeChannel()
        manager = FakeManager()
        payload = DeploymentRequest(
            job_id="deploy-1",
            model_version="training-final-004",
        )

        event = process_deployment_message(channel, manager, payload)
        status_events = [
            item for item in channel.published
            if item["routing_key"] == "app.training"
        ]
        sse_events = [
            item for item in channel.published
            if item["exchange"] == SSE_EXCHANGE
        ]

        assert manager.calls == [
            ("preload", "training-final-004"),
            ("validate", None),
            ("switch", None),
        ]
        assert [item["body"]["status"] for item in status_events] == [
            "RUNNING",
            "RUNNING",
            "RUNNING",
            "COMPLETED",
        ]
        assert all("job_id" in item["body"] for item in status_events)
        assert all(item["body"]["job_type"] == "model" for item in status_events)
        assert all("stage" in item["body"] for item in status_events)
        assert all("model_version" in item["body"] for item in status_events)
        assert all("error_message" in item["body"] for item in status_events)
        assert all("finished_at" in item["body"] for item in status_events)
        assert [item["body"].get("stage") for item in status_events[:3]] == [
            "PRELOAD",
            "VALIDATE",
            "SWITCH",
        ]
        assert {item["exchange"] for item in status_events} == {AI2APP_EXCHANGE}
        assert {item["routing_key"] for item in status_events} == {"app.training"}
        assert len(sse_events) == 1
        assert sse_events[0]["routing_key"] == ""
        assert sse_events[0]["body"]["data"] == SSE_SUCCESS_MESSAGE
        assert event["active_model_version"] == "training-final-004"
        assert event["job_type"] == "model"

    def test_sse_publish_failure_does_not_fail_completed_deployment(self):
        class FakeChannel:
            def __init__(self):
                self.published = []

            def basic_publish(self, **kwargs):
                if kwargs["exchange"] == SSE_EXCHANGE:
                    raise RuntimeError("sse down")
                self.published.append(
                    {
                        "exchange": kwargs["exchange"],
                        "routing_key": kwargs["routing_key"],
                        "body": json.loads(kwargs["body"].decode("utf-8")),
                    }
                )

        class FakeManager:
            def preload(self, _version):
                pass

            def validate(self):
                pass

            def switch(self):
                return {"status": "switched", "model_version": "training-final-004"}

        channel = FakeChannel()
        payload = DeploymentRequest(job_id="deploy-1", model_version="training-final-004")

        event = process_deployment_message(channel, FakeManager(), payload)

        assert event["status"] == "COMPLETED"
        assert [item["body"]["status"] for item in channel.published] == [
            "RUNNING",
            "RUNNING",
            "RUNNING",
            "COMPLETED",
        ]

    def test_process_deployment_without_model_version_uses_latest_preload_path(self):
        class FakeChannel:
            def __init__(self):
                self.published = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

        class FakeManager:
            def __init__(self):
                self.calls = []

            def preload(self, version):
                self.calls.append(("preload", version))
                return {"status": "preloaded", "model_version": "latest-from-json"}

            def validate(self):
                self.calls.append(("validate", None))

            def switch(self):
                self.calls.append(("switch", None))
                return {"status": "switched", "model_version": "latest-from-json"}

        channel = FakeChannel()
        manager = FakeManager()
        payload = DeploymentRequest(user_id=1, task_type="deployment")

        event = process_deployment_message(channel, manager, payload)

        assert manager.calls[0] == ("preload", None)
        assert event["model_version"] is None
        assert event["job_type"] == "model"
        assert event["active_model_version"] == "latest-from-json"
        assert channel.published[0]["job_id"].startswith("deployment-1-")
        assert all(item["job_type"] == "model" for item in channel.published if "status" in item)

    def test_validate_failure_does_not_switch(self):
        class FakeChannel:
            def basic_publish(self, **kwargs):
                pass

        class FakeManager:
            def __init__(self):
                self.switched = False

            def preload(self, version):
                return {"status": "preloaded", "model_version": version}

            def validate(self):
                raise RuntimeError("validation failed")

            def switch(self):
                self.switched = True
                return {"status": "switched", "model_version": "bad"}

        manager = FakeManager()
        payload = DeploymentRequest(job_id="deploy-1", model_version="training-final-004")

        with pytest.raises(Exception, match="validation failed"):
            process_deployment_message(FakeChannel(), manager, payload)

        assert manager.switched is False

    def test_deployment_callback_accepts_model_version_camel_case_and_acks(self):
        class FakeMethod:
            delivery_tag = 11
            routing_key = "2ai.deployment"
            exchange = "x.app2ai.direct"
            redelivered = False

        class FakeChannel:
            def __init__(self):
                self.published = []
                self.acks = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

            def basic_ack(self, delivery_tag):
                self.acks.append(delivery_tag)

        class FakeManager:
            def __init__(self):
                self.calls = []

            def preload(self, version):
                self.calls.append(("preload", version))

            def validate(self):
                self.calls.append(("validate", None))

            def switch(self):
                self.calls.append(("switch", None))
                return {"status": "switched", "model_version": "training-final-004"}

        manager = FakeManager()
        runner = DeploymentConsumerRunner(manager)
        channel = FakeChannel()
        body = json.dumps(
            {
                "jobId": "deploy-1",
                "modelVersion": "training-final-004",
            }
        ).encode("utf-8")

        runner._callback(channel, FakeMethod(), None, body)

        assert manager.calls == [
            ("preload", "training-final-004"),
            ("validate", None),
            ("switch", None),
        ]
        assert channel.acks == [11]
        status_events = [event for event in channel.published if "stage" in event]
        assert [event["stage"] for event in status_events] == [
            "PRELOAD",
            "VALIDATE",
            "SWITCH",
            "COMPLETED",
        ]

    def test_deployment_callback_accepts_admin_payload_and_generates_job_id(self):
        class FakeMethod:
            delivery_tag = 12
            routing_key = "2ai.deployment"
            exchange = "x.app2ai.direct"
            redelivered = False

        class FakeChannel:
            def __init__(self):
                self.published = []
                self.acks = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

            def basic_ack(self, delivery_tag):
                self.acks.append(delivery_tag)

        class FakeManager:
            def __init__(self):
                self.calls = []

            def preload(self, version):
                self.calls.append(("preload", version))

            def validate(self):
                self.calls.append(("validate", None))

            def switch(self):
                self.calls.append(("switch", None))
                return {"status": "switched", "model_version": "latest-from-json"}

        manager = FakeManager()
        runner = DeploymentConsumerRunner(manager)
        channel = FakeChannel()
        body = json.dumps({"user_id": 1, "task_type": "deployment"}).encode("utf-8")

        runner._callback(channel, FakeMethod(), None, body)

        assert manager.calls[0] == ("preload", None)
        assert channel.acks == [12]
        assert channel.published[0]["job_id"].startswith("deployment-1-")
        completed = [event for event in channel.published if event.get("status") == "COMPLETED"][0]
        assert completed["active_model_version"] == "latest-from-json"
        assert completed["job_type"] == "model"
        assert channel.published[-1]["data"] == SSE_SUCCESS_MESSAGE

    def test_deployment_callback_accepts_admin_model_job_payload(self):
        class FakeMethod:
            delivery_tag = 14
            routing_key = "2ai.deployment"
            exchange = "x.app2ai.direct"
            redelivered = False

        class FakeChannel:
            def __init__(self):
                self.published = []
                self.acks = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

            def basic_ack(self, delivery_tag):
                self.acks.append(delivery_tag)

        class FakeManager:
            def __init__(self):
                self.calls = []

            def preload(self, version):
                self.calls.append(("preload", version))

            def validate(self):
                self.calls.append(("validate", None))

            def switch(self):
                self.calls.append(("switch", None))
                return {"status": "switched", "model_version": "latest-from-json"}

        manager = FakeManager()
        runner = DeploymentConsumerRunner(manager)
        channel = FakeChannel()
        body = json.dumps(
            {"user_id": 1, "task_type": "deployment", "job_type": "model"}
        ).encode("utf-8")

        runner._callback(channel, FakeMethod(), None, body)

        assert manager.calls == [
            ("preload", None),
            ("validate", None),
            ("switch", None),
        ]
        assert channel.acks == [14]
        status_events = [event for event in channel.published if "status" in event]
        assert all(event["job_type"] == "model" for event in status_events)
        completed = [event for event in channel.published if event.get("status") == "COMPLETED"][0]
        assert completed["status"] == "COMPLETED"
        assert completed["job_type"] == "model"
        assert channel.published[-1]["data"] == SSE_SUCCESS_MESSAGE

    def test_deployment_callback_stage_failure_publishes_failed_with_job_type(self):
        class FakeMethod:
            delivery_tag = 16
            routing_key = "2ai.deployment"
            exchange = "x.app2ai.direct"
            redelivered = False

        class FakeChannel:
            def __init__(self):
                self.published = []
                self.acks = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

            def basic_ack(self, delivery_tag):
                self.acks.append(delivery_tag)

        class FakeManager:
            def preload(self, _version):
                pass

            def validate(self):
                raise RuntimeError("validation failed")

            def switch(self):
                raise AssertionError("switch must not run after validation failure")

        runner = DeploymentConsumerRunner(FakeManager())
        channel = FakeChannel()
        body = json.dumps(
            {
                "job_id": "deploy-1",
                "model_version": "training-final-004",
                "job_type": "model",
            }
        ).encode("utf-8")

        runner._callback(channel, FakeMethod(), None, body)

        assert channel.acks == [16]
        failed = [event for event in channel.published if event.get("status") == "FAILED"][0]
        assert failed["job_type"] == "model"
        assert failed["stage"] == "VALIDATE"
        assert failed["error_message"] == "validation failed"
        assert channel.published[-1]["data"] == SSE_FAILURE_MESSAGE

    def test_deployment_callback_invalid_task_type_publishes_failed_and_acks(self):
        class FakeMethod:
            delivery_tag = 13
            routing_key = "2ai.deployment"
            exchange = "x.app2ai.direct"
            redelivered = False

        class FakeChannel:
            def __init__(self):
                self.published = []
                self.acks = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

            def basic_ack(self, delivery_tag):
                self.acks.append(delivery_tag)

        class FakeManager:
            def preload(self, _version):
                raise AssertionError("preload must not run for invalid task_type")

        runner = DeploymentConsumerRunner(FakeManager())
        channel = FakeChannel()
        body = json.dumps({"user_id": 1, "task_type": "collect"}).encode("utf-8")

        runner._callback(channel, FakeMethod(), None, body)

        assert channel.acks == [13]
        status_events = [event for event in channel.published if event.get("status") == "FAILED"]
        assert len(status_events) == 1
        failed = status_events[0]
        assert failed["job_id"].startswith("deployment-1-")
        assert failed["job_type"] == "model"
        assert failed["status"] == "FAILED"
        assert failed["model_version"] is None
        assert failed["stage"] == "PARSE"
        assert failed["error_message"] == "Unsupported deployment task_type: collect"
        assert failed["finished_at"]
        assert channel.published[-1]["data"] == SSE_FAILURE_MESSAGE

    def test_deployment_callback_invalid_job_type_publishes_failed_and_acks(self):
        class FakeMethod:
            delivery_tag = 15
            routing_key = "2ai.deployment"
            exchange = "x.app2ai.direct"
            redelivered = False

        class FakeChannel:
            def __init__(self):
                self.published = []
                self.acks = []

            def basic_publish(self, **kwargs):
                self.published.append(json.loads(kwargs["body"].decode("utf-8")))

            def basic_ack(self, delivery_tag):
                self.acks.append(delivery_tag)

        class FakeManager:
            def preload(self, _version):
                raise AssertionError("preload must not run for invalid job_type")

        runner = DeploymentConsumerRunner(FakeManager())
        channel = FakeChannel()
        body = json.dumps(
            {"user_id": 1, "task_type": "deployment", "job_type": "dataset"}
        ).encode("utf-8")

        runner._callback(channel, FakeMethod(), None, body)

        assert channel.acks == [15]
        status_events = [event for event in channel.published if event.get("status") == "FAILED"]
        assert len(status_events) == 1
        failed = status_events[0]
        assert failed["job_id"].startswith("deployment-1-")
        assert failed["job_type"] == "dataset"
        assert failed["status"] == "FAILED"
        assert failed["model_version"] is None
        assert failed["stage"] == "PARSE"
        assert failed["error_message"] == "Unsupported deployment job_type: dataset"
        assert failed["finished_at"]
        assert channel.published[-1]["data"] == SSE_FAILURE_MESSAGE
