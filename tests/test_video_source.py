"""Tests de VideoSource: apertura, lectura, metadatos y fin de flujo.

Se genera un vídeo sintético mínimo con OpenCV para no depender de ficheros externos.
"""

import cv2
import numpy as np
import pytest

from capture.video_source import VideoSource, VideoSourceError, EndOfStream


FPS = 10.0
N_FRAMES = 20


@pytest.fixture
def video(tmp_path):
    """Crea un clip sintético de 20 frames a 10 fps y devuelve su ruta."""
    ruta = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return str(ruta)


def test_lee_frames_y_devuelve_imagenes(video):
    with VideoSource(video) as source:
        frame = source.read()
    assert frame.shape == (48, 64, 3)


def test_frame_count_y_fps(video):
    with VideoSource(video) as source:
        assert source.frame_count() == N_FRAMES
        assert source.fps() == pytest.approx(FPS, rel=0.1)


def test_al_agotar_el_video_se_señala_fin_de_flujo(video):
    with VideoSource(video) as source:
        for _ in range(N_FRAMES):
            source.read()
        with pytest.raises(EndOfStream):
            source.read()


def test_fuente_inexistente_lanza_error(tmp_path):
    with pytest.raises(VideoSourceError):
        VideoSource(str(tmp_path / "no_existe.mp4")).open()


def test_leer_sin_abrir_lanza_error(video):
    with pytest.raises(VideoSourceError):
        VideoSource(video).read()


def test_metadatos_sin_abrir_devuelven_cero(video):
    # Sin fuente abierta no hay metadatos: se devuelve 0 en lugar de fallar,
    # que es también el caso de webcams y streams donde no se conocen.
    source = VideoSource(video)
    assert source.frame_count() == 0
    assert source.fps() == 0.0
