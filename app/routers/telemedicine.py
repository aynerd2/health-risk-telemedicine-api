"""
Telemedicine module (Section 3.2.8). Real-time chat is handled here over a
WebSocket, one connection per appointment room. Video calling is handled
client-side in Next.js via a managed WebRTC provider (Daily/Twilio/Agora) and
does not route media through this backend.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/v1/telemedicine", tags=["Telemedicine"])


class ConsultationRoom:
    """In-memory registry of active WebSocket connections per appointment.
    For a multi-instance production deployment, back this with Redis pub/sub."""

    def __init__(self) -> None:
        self.rooms: dict[int, list[WebSocket]] = {}

    async def connect(self, appt_id: int, websocket: WebSocket) -> None:
        await websocket.accept()
        self.rooms.setdefault(appt_id, []).append(websocket)

    def disconnect(self, appt_id: int, websocket: WebSocket) -> None:
        if appt_id in self.rooms and websocket in self.rooms[appt_id]:
            self.rooms[appt_id].remove(websocket)

    async def broadcast(self, appt_id: int, sender: str, message: str) -> None:
        for connection in self.rooms.get(appt_id, []):
            await connection.send_json({"sender": sender, "message": message})


rooms = ConsultationRoom()


@router.websocket("/ws/{appt_id}")
async def consultation_chat(websocket: WebSocket, appt_id: int, sender: str = "user"):
    """
    Connect from the Next.js consultation page with:
        new WebSocket(`wss://<api-host>/api/v1/telemedicine/ws/${apptId}?sender=${role}`)
    A JWT should also be validated here in production (e.g. via a query param
    or subprotocol) before accepting the connection.
    """
    await rooms.connect(appt_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await rooms.broadcast(appt_id, sender, data)
    except WebSocketDisconnect:
        rooms.disconnect(appt_id, websocket)
