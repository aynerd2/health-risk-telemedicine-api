from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.core.database import get_session
from app.core.deps import require_role
from app.models.entities import HealthRecord, Patient, Prediction, User, UserRole
from app.schemas.prediction import HealthIntakeRequest, HealthRecordHistory, PredictionResponse
from app.services.prediction_service import predict_all

router = APIRouter(prefix="/api/v1/predictions", tags=["AI Risk Prediction"])


@router.post("", response_model=PredictionResponse)
def submit_intake_and_predict(
    payload: HealthIntakeRequest,
    current_user: User = Depends(require_role(UserRole.patient)),
    session: Session = Depends(get_session),
):
    """
    Implements the workflow in Figure 3.2: a patient submits the health-intake
    form, the system pre-processes the input, and the three Logistic Regression
    models each return a risk indication.
    """
    patient = session.exec(select(Patient).where(Patient.user_id == current_user.user_id)).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    record = HealthRecord(patient_id=patient.patient_id, **payload.model_dump())
    session.add(record)
    session.commit()
    session.refresh(record)

    try:
        results = predict_all(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for r in results:
        session.add(
            Prediction(
                record_id=record.record_id,
                condition=r.condition,
                risk_score=r.risk_score,
                risk_class=r.risk_class,
            )
        )
    session.commit()

    return PredictionResponse(
        record_id=record.record_id,
        results=results,
        any_elevated=any(r.risk_class.value == "elevated" for r in results),
    )


@router.get("/mine", response_model=list[HealthRecordHistory])
def my_prediction_history(
    current_user: User = Depends(require_role(UserRole.patient)),
    session: Session = Depends(get_session),
):
    """Powers the patient's history page: every past intake plus the risk
    indications generated from it, most recent first."""
    patient = session.exec(select(Patient).where(Patient.user_id == current_user.user_id)).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    records = session.exec(
        select(HealthRecord)
        .where(HealthRecord.patient_id == patient.patient_id)
        .order_by(HealthRecord.date_recorded.desc())
    ).all()

    history = []
    for record in records:
        predictions = session.exec(select(Prediction).where(Prediction.record_id == record.record_id)).all()
        history.append(HealthRecordHistory(record_id=record.record_id, date_recorded=record.date_recorded, predictions=predictions))
    return history
