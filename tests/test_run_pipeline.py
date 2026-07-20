"""Tests de las utilidades de run_pipeline.

Cubren el formateo del instante del vídeo (`mm:ss` / `hh:mm:ss`) con el que se etiquetan
los eventos, en lugar de la hora del sistema.
"""

from run_pipeline import _format_video_time


def test_segundos_se_formatean_como_mm_ss():
    assert _format_video_time(0) == "00:00"
    assert _format_video_time(5) == "00:05"
    assert _format_video_time(65) == "01:05"


def test_se_trunca_a_segundos_enteros():
    # 8.96 s pertenece al segundo 8: no se redondea hacia arriba.
    assert _format_video_time(8.96) == "00:08"


def test_videos_de_mas_de_una_hora_incluyen_horas():
    assert _format_video_time(3600) == "01:00:00"
    assert _format_video_time(3725) == "01:02:05"


def test_por_debajo_de_una_hora_no_se_muestran_horas():
    # 59:59 sigue en formato mm:ss; a partir de 1 h cambia el formato.
    assert _format_video_time(3599) == "59:59"
