from datetime import datetime

from api.services import summarize_service
from api.services.summarize_service import parse_datetime_kst


def test_parse_datetime_kst_relative_korean_expression():
    base_datetime = datetime.fromisoformat("2026-04-18T09:00:00+09:00")

    date, time = parse_datetime_kst("다음주 화요일", "오후 2시", base_datetime)

    assert date == "2026-04-21"
    assert time == "14:00"


def test_parse_datetime_kst_absolute_month_day():
    base_datetime = datetime.fromisoformat("2026-04-18T09:00:00+09:00")

    date, time = parse_datetime_kst("3월 28일", "오전 10시", base_datetime)

    assert date == "2026-03-28"
    assert time == "10:00"


def test_parse_datetime_kst_logs_warning_when_unparseable(monkeypatch):
    base_datetime = datetime.fromisoformat("2026-04-18T09:00:00+09:00")
    captured = {}

    def _fake_warning(msg, **fields):
        captured["msg"] = msg
        captured["fields"] = fields

    monkeypatch.setattr(summarize_service.log, "warning", _fake_warning)

    date, time = parse_datetime_kst("언젠가", "적당한 시간", base_datetime)

    assert date is None
    assert time is None
    assert captured["msg"] == "schedule_datetime_parse_failed"
    assert captured["fields"]["date_text"] == "언젠가"
    assert captured["fields"]["time_text"] == "적당한 시간"
    assert captured["fields"]["joined_text"] == "언젠가 적당한 시간"
    assert captured["fields"]["timezone"] == "Asia/Seoul"
