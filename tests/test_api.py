"""Tests de la API REST.

Se ejercitan los endpoints con el TestClient de FastAPI, sustituyendo por dobles las
dependencias con efectos externos: la base de datos (SQLite en memoria), el chat de Q&A
(que llamaría a Groq) y el arranque del pipeline (que lanzaría YOLO en un hilo).
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api as api_module
from database.event_store import EventStore


class QAChainFalso:
    """Doble del chat de Q&A: registra la última llamada y devuelve una respuesta fija."""

    def __init__(self):
        self.ultima_pregunta = None
        self.ultimos_eventos = None

    def ask(self, question, events):
        self.ultima_pregunta = question
        self.ultimos_eventos = events
        return "respuesta de prueba"


@pytest.fixture
def store():
    return EventStore(db_path=":memory:")


@pytest.fixture
def qa():
    return QAChainFalso()


@pytest.fixture
def client(monkeypatch, tmp_path, store, qa):
    """TestClient con todas las dependencias externas sustituidas."""
    monkeypatch.setattr(api_module, "event_store", store)
    monkeypatch.setattr(api_module, "qa_chain", qa)
    monkeypatch.setattr(api_module, "current_video", [None])
    monkeypatch.setattr(api_module, "latest_frame", [None])
    monkeypatch.setattr(api_module, "pipeline_thread", None)
    monkeypatch.setattr(api_module, "UPLOADS_DIR", tmp_path)
    # Evita lanzar el pipeline real (YOLO + hilo) al llamar a /start.
    monkeypatch.setattr(api_module, "iniciar_pipeline", lambda **kwargs: None)
    with TestClient(api_module.app) as c:
        yield c


def _add(store, track_id, video, risk="high"):
    store.add_event(
        track_id=track_id,
        alert=f"alerta {track_id}",
        risk_level=risk,
        timestamp="00:12",
        frame_path=f"/frames/{track_id}.jpg",
        video=video,
    )


# --- Consulta de eventos y estado ---------------------------------------------------

def test_events_vacio_al_principio(client):
    respuesta = client.get("/events")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"events": []}


def test_events_devuelve_los_eventos_guardados(client, store):
    _add(store, "1", "A.mp4")
    eventos = client.get("/events").json()["events"]
    assert len(eventos) == 1
    assert eventos[0]["track_id"] == "1"
    assert eventos[0]["timestamp"] == "00:12"


def test_events_filtra_por_el_video_actual(client, store):
    _add(store, "1", "A.mp4")
    _add(store, "2", "B.mp4")
    client.post("/start", json={"path": "/videos/A.mp4"})

    eventos = client.get("/events").json()["events"]

    assert [e["track_id"] for e in eventos] == ["1"]


def test_status_detenido_y_total_de_eventos(client, store):
    _add(store, "1", "A.mp4")
    _add(store, "2", "B.mp4")

    cuerpo = client.get("/status").json()

    assert cuerpo["status"] == "stopped"
    assert cuerpo["total_events"] == 2


def test_progress_devuelve_el_avance(client):
    cuerpo = client.get("/progress").json()
    assert set(cuerpo) == {"processed", "total", "percent"}


# --- Último frame --------------------------------------------------------------------

def test_latest_frame_sin_frame_devuelve_204(client):
    assert client.get("/latest_frame").status_code == 204


def test_latest_frame_devuelve_jpeg(client, monkeypatch):
    monkeypatch.setattr(api_module, "latest_frame", [np.zeros((8, 8, 3), dtype=np.uint8)])

    respuesta = client.get("/latest_frame")

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"] == "image/jpeg"
    # El contenido debe ser un JPEG decodificable.
    assert cv2.imdecode(np.frombuffer(respuesta.content, np.uint8), cv2.IMREAD_COLOR) is not None


# --- Preguntas sobre la escena -------------------------------------------------------

def test_ask_delega_en_el_qa_chain(client, store, qa):
    _add(store, "1", "A.mp4")

    respuesta = client.post("/ask", json={"text": "¿Qué ha pasado?"})

    assert respuesta.json() == {"answer": "respuesta de prueba"}
    assert qa.ultima_pregunta == "¿Qué ha pasado?"
    assert len(qa.ultimos_eventos) == 1


# --- Subida de vídeos ----------------------------------------------------------------

def test_upload_guarda_el_fichero(client, tmp_path):
    respuesta = client.post("/upload_video", files={"file": ("clip.mp4", b"contenido", "video/mp4")})

    ruta = respuesta.json()["video_path"]
    assert (tmp_path / "clip.mp4").read_bytes() == b"contenido"
    assert ruta == str(tmp_path / "clip.mp4")


def test_upload_neutraliza_rutas_maliciosas(client, tmp_path):
    """Un nombre con ../ no debe escribir fuera del directorio de uploads."""
    respuesta = client.post(
        "/upload_video",
        files={"file": ("../../../evil.mp4", b"x", "video/mp4")},
    )

    ruta = respuesta.json()["video_path"]
    assert ruta == str(tmp_path / "evil.mp4")
    assert (tmp_path / "evil.mp4").exists()


# --- Control del pipeline ------------------------------------------------------------

def test_start_fija_el_video_actual_y_resetea_el_progreso(client):
    api_module.progress.update({"processed": 99, "total": 99, "percent": 100.0})

    respuesta = client.post("/start", json={"path": "/videos/A.mp4"})

    assert respuesta.json() == {"status": "started"}
    assert api_module.current_video[0] == "A.mp4"
    assert api_module.progress == {"processed": 0, "total": 0, "percent": 0.0}


def test_stop_señaliza_la_parada(client):
    api_module.stop_event.clear()

    respuesta = client.post("/stop")

    assert respuesta.json() == {"status": "stopped"}
    assert api_module.stop_event.is_set()


def test_clear_events_borra_solo_el_video_actual(client, store):
    _add(store, "1", "A.mp4")
    _add(store, "2", "B.mp4")
    client.post("/start", json={"path": "/videos/A.mp4"})

    respuesta = client.post("/clear_events")

    assert respuesta.json() == {"status": "cleared"}
    restantes = store.get_all_events()
    assert [e["video"] for e in restantes] == ["B.mp4"]
    # Deja de haber vídeo activo tras limpiar.
    assert api_module.current_video[0] is None
