"""
Re-score stored Prediction rows from their HealthRecord using the current
prediction_service (same code path as POST /api/v1/predictions).

Needed after the October 2026 prediction_service fixes: the heart model's
inverted target and recoded cp/restecg/slope/thal, blank fields bypassing
imputation, and the hypertension smoking flag. Every Prediction stored
before those fixes holds a score computed the old (wrong) way.

Dry run by default — prints what would change and writes nothing:

    python scripts/rescore_predictions.py --conditions heart_disease diabetes hypertension

With --apply, BEFORE touching any row it (1) copies the affected rows into a
new table prediction_backup_<UTC timestamp> in the same database and (2)
writes them to rescore_backups/<same name>.json (gitignored — it is patient
health data; don't commit or share it). Then it updates risk_score and
risk_class in place (date_generated is kept).

To undo an applied run:

    python scripts/rescore_predictions.py --restore prediction_backup_<timestamp>

Uses DATABASE_URL from the environment / .env, like the app itself.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from app.core.database import engine  # noqa: E402
from app.models.entities import Condition, HealthRecord, Prediction  # noqa: E402
from app.schemas.prediction import HealthIntakeRequest  # noqa: E402
from app.services import prediction_service  # noqa: E402

BACKUP_DIR = BACKEND_DIR / "rescore_backups"
BACKUP_TABLE_RE = re.compile(r"^prediction_backup_\d{8}T\d{6}Z$")


def intake_from_record(record: HealthRecord) -> HealthIntakeRequest:
    # The POST handler builds HealthRecord from HealthIntakeRequest.model_dump(),
    # so every intake field is a HealthRecord column of the same name.
    return HealthIntakeRequest.model_validate({name: getattr(record, name) for name in HealthIntakeRequest.model_fields})


def rescore(conditions: list[Condition], apply: bool) -> None:
    prediction_service.load_models()

    with Session(engine) as session:
        rows = session.exec(
            select(Prediction, HealthRecord)
            .join(HealthRecord, HealthRecord.record_id == Prediction.record_id)
            .where(Prediction.condition.in_(conditions))
            .order_by(Prediction.prediction_id)
        ).all()

        changes = []
        for prediction, record in rows:
            new = prediction_service.predict_one(prediction.condition, intake_from_record(record))
            changes.append((prediction, new))

        print(f"{len(rows)} prediction rows for {[c.value for c in conditions]}")
        for condition in conditions:
            subset = [(p, n) for p, n in changes if p.condition == condition]
            flipped = [(p, n) for p, n in subset if p.risk_class != n.risk_class]
            moved = [(p, n) for p, n in subset if abs(p.risk_score - n.risk_score) >= 0.0001]
            print(f"  {condition.value:13s} rows={len(subset):5d}  score changed={len(moved):5d}  risk_class flipped={len(flipped):5d}")
            for p, n in flipped[:10]:
                print(
                    f"    prediction_id={p.prediction_id} record_id={p.record_id}: "
                    f"{p.risk_score:.4f} {p.risk_class.value} -> {n.risk_score:.4f} {n.risk_class.value}"
                )
            if len(flipped) > 10:
                print(f"    ... and {len(flipped) - 10} more")

        if not apply:
            print("\nDry run — nothing written. Re-run with --apply to back up and update.")
            return
        if not changes:
            print("\nNothing to update.")
            return

        backup_table = f"prediction_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        ids = [p.prediction_id for p, _ in changes]

        BACKUP_DIR.mkdir(exist_ok=True)
        backup_file = BACKUP_DIR / f"{backup_table}.json"
        backup_file.write_text(
            json.dumps(
                [
                    {
                        "prediction_id": p.prediction_id,
                        "record_id": p.record_id,
                        "condition": p.condition.value,
                        "risk_score": p.risk_score,
                        "risk_class": p.risk_class.value,
                        "date_generated": p.date_generated.isoformat(),
                    }
                    for p, _ in changes
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        # A real table in the same DB, so a restore doesn't depend on the
        # machine this script happened to run on still having the JSON file.
        id_list = ",".join(str(i) for i in ids)  # integers from our own query — safe to inline
        # MySQL hosts like Aiven set sql_require_primary_key, which rejects a
        # plain CREATE TABLE ... AS SELECT; MySQL can declare the key inline,
        # SQLite can't parse that form.
        primary_key = " (PRIMARY KEY (prediction_id))" if engine.dialect.name == "mysql" else ""
        session.exec(text(f"CREATE TABLE {backup_table}{primary_key} AS SELECT * FROM prediction WHERE prediction_id IN ({id_list})"))
        session.commit()
        backed_up = session.exec(text(f"SELECT COUNT(*) FROM {backup_table}")).one()[0]
        if backed_up != len(ids):
            raise SystemExit(f"Backup table has {backed_up} rows, expected {len(ids)} — aborting before any update.")
        print(f"\nBacked up {backed_up} rows to table {backup_table} and {backup_file}")

        for prediction, new in changes:
            prediction.risk_score = new.risk_score
            prediction.risk_class = new.risk_class
            session.add(prediction)
        session.commit()
        print(f"Updated {len(changes)} rows. Undo with: python scripts/rescore_predictions.py --restore {backup_table}")


def restore(backup_table: str) -> None:
    if not BACKUP_TABLE_RE.match(backup_table):
        raise SystemExit(f"Not a backup table name produced by this script: {backup_table!r}")
    with Session(engine) as session:
        rows = session.exec(text(f"SELECT prediction_id, risk_score, risk_class FROM {backup_table}")).all()
        for prediction_id, risk_score, risk_class in rows:
            session.exec(
                text("UPDATE prediction SET risk_score = :s, risk_class = :c WHERE prediction_id = :i"),
                params={"s": risk_score, "c": risk_class, "i": prediction_id},
            )
        session.commit()
    print(f"Restored {len(rows)} rows from {backup_table}. The backup table itself is left in place.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", choices=[c.value for c in Condition], default=[Condition.heart_disease.value])
    parser.add_argument("--apply", action="store_true", help="back up, then write the new scores")
    parser.add_argument("--restore", metavar="BACKUP_TABLE", help="undo an applied run from its backup table")
    args = parser.parse_args()

    if args.restore:
        restore(args.restore)
    else:
        rescore([Condition(c) for c in args.conditions], apply=args.apply)


if __name__ == "__main__":
    main()
