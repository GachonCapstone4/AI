import csv
import importlib.util
import sys
import types


def _stub_missing_batch_dependencies():
    if importlib.util.find_spec("boto3") is None:
        sys.modules["boto3"] = types.ModuleType("boto3")

    if importlib.util.find_spec("mysql") is None:
        mysql = types.ModuleType("mysql")
        mysql.connector = types.ModuleType("mysql.connector")
        sys.modules["mysql"] = mysql
        sys.modules["mysql.connector"] = mysql.connector

    if importlib.util.find_spec("pika") is None:
        pika = types.ModuleType("pika")
        pika.PlainCredentials = object
        pika.ConnectionParameters = object
        pika.BlockingConnection = object
        pika.BasicProperties = object
        sys.modules["pika"] = pika


def test_create_csv_adds_regenerated_email_text(tmp_path):
    _stub_missing_batch_dependencies()
    from batch.dataset_batch import create_csv

    csv_path = tmp_path / "dataset.csv"
    rows = [
        {
            "emailId": "train_1",
            "threadId": "thread-1",
            "from": "sender@example.com",
            "subject": "Invoice request",
            "body": "Please send the invoice.",
            "email_text": "stale value",
            "domain": "Finance",
            "intent": "Invoice Request",
        },
        {
            "emailId": "train_2",
            "threadId": None,
            "from": "empty@example.com",
            "subject": None,
            "body": "",
            "domain": "General",
            "intent": "Question",
        },
    ]

    create_csv(rows, str(csv_path))

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        written_rows = list(reader)

    assert reader.fieldnames == [
        "emailId",
        "threadId",
        "from",
        "subject",
        "body",
        "email_text",
        "domain",
        "intent",
    ]
    assert written_rows[0]["email_text"] == "Invoice request\nPlease send the invoice."
    assert written_rows[1]["email_text"] == ""
