"""Mirror completed study sessions to a private, dedicated Google Sheet tab."""
import argparse
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from statistics import mean

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials
from requests.exceptions import RequestException

from user_study.backend.service import results


TAB = "Completed results"
METRICS = (
    ("detected", "Falls detected"),
    ("missed", "Falls missed"),
    ("false_alarms", "False alarms"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("mean_latency", "Mean detection delay (s)"),
)
RATINGS = (("confidence", "Confidence (1–5)"), ("effort", "Effort (1–5)"), ("ease", "Ease (1–5)"))
HEADERS = [
    "Participant ID", "Group", "Age range", "Monitoring experience", "Round order",
    *(f"{label} · {column}" for label in ("Manual", "Automated") for _, column in (*METRICS, *RATINGS)),
    "Mean alert acknowledgment (s)", "Trust in alerts (1–5)", "Participant feedback",
]


def completed_rows(database):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)) as db:
        sessions = (json.loads(body) for (body,) in db.execute("SELECT body FROM sessions ORDER BY rowid"))
        rows = []
        for session in sessions:
            if session["stage"] != "completed" or session["manifest"]["mode"] != "study":
                continue
            record = results(session)
            runs = {run["condition"]: run for run in record["runs"] if run["completed"]}
            if runs.keys() != {"manual", "automated"} or not record["survey"]:
                continue
            survey = record["survey"]
            row = [
                record["code"], record["group"], record["age_range"], record["experience"],
                ", ".join(run["condition"].title() for run in record["runs"]),
            ]
            for condition in ("manual", "automated"):
                row.extend(runs[condition]["metrics"][key] for key, _ in METRICS)
                row.extend(survey.get(f"{condition}_{key}") for key, _ in RATINGS)
            acknowledgments = runs["automated"]["acknowledgment_latencies"]
            row.extend([mean(acknowledgments) if acknowledgments else None, survey.get("trust"), survey.get("feedback")])
            rows.append([value if value is not None else "" for value in row])
    return rows


def cell(value):
    if value == "":
        return {}
    field = "numberValue" if isinstance(value, (int, float)) else "stringValue"
    return {"userEnteredValue": {field: value}}


def sync(client, url, database):
    response = client.get(f"{url}/values/'{TAB}'!A1:Z", params={"valueRenderOption": "UNFORMATTED_VALUE"}, timeout=20)
    response.raise_for_status()
    header, *existing = response.json()["values"]
    if header != HEADERS:
        raise ValueError("The Completed results sheet headers do not match the study export.")
    rows = completed_rows(database)
    normalized = [row + [""] * (len(HEADERS) - len(row)) for row in existing]
    if normalized == rows:
        return len(rows)

    response = client.get(url, params={"fields": "sheets.properties(sheetId,title,gridProperties)"}, timeout=20)
    response.raise_for_status()
    sheet = next(item["properties"] for item in response.json()["sheets"] if item["properties"]["title"] == TAB)
    end = max(len(existing), len(rows)) + 1
    requests = []
    capacity = sheet["gridProperties"]["rowCount"]
    if end > capacity:
        requests.append({"appendDimension": {"sheetId": sheet["sheetId"], "dimension": "ROWS", "length": end - capacity}})
    requests.append({"updateCells": {
        "range": {"sheetId": sheet["sheetId"], "startRowIndex": 1, "endRowIndex": end,
                  "startColumnIndex": 0, "endColumnIndex": len(HEADERS)},
        "rows": [{"values": [cell(value) for value in row]} for row in rows],
        "fields": "userEnteredValue",
    }})
    response = client.post(f"{url}:batchUpdate", json={"requests": requests}, timeout=30)
    response.raise_for_status()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--spreadsheet", required=True)
    parser.add_argument("--watch", action="store_true", help="Sync every 30 seconds until stopped.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    credentials = Credentials.from_service_account_file(
        str(args.credentials), scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{args.spreadsheet}"
    previous = None
    with AuthorizedSession(credentials) as client:
        while True:
            try:
                count = sync(client, url, args.database)
            except (RequestException, sqlite3.Error) as error:
                if not args.watch:
                    raise
                logging.warning("Sync failed, will retry (%s).", type(error).__name__)
            else:
                if count != previous:
                    logging.info("Sheet synchronized with %s completed study sessions.", count)
                    previous = count
            if not args.watch:
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
