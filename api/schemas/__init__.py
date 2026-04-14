"""
스키마 분리 모듈의 호환 계층.

기존 `from api.schemas import ...` import 를 유지하면서,
분류/요약/학습 경로의 스키마를 파일 단위로 분리한다.
"""

from api.schemas.classify import Classification, ClassifyRequest, ClassifyResponse
from api.schemas.common import ErrorResponse, ResponseMeta, SummarizeResponse
from api.schemas.training import TrainingJobRequest, TrainingJobResult, TrainingMetrics

__all__ = [
    "Classification",
    "ClassifyRequest",
    "ClassifyResponse",
    "ErrorResponse",
    "ResponseMeta",
    "SummarizeResponse",
    "TrainingJobRequest",
    "TrainingJobResult",
    "TrainingMetrics",
]
