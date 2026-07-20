"""Tests del agente de alerta (grafo LangGraph).

Verifican la regla de negocio central: solo los riesgos `medium` y `high` generan un mensaje
de alerta, mientras que `low` se ignora. Las dos llamadas al LLM (descripción de escena y
evaluación de riesgo) se sustituyen por dobles, de modo que no hay tráfico de red.
"""

import pytest

import ai.alert_agent as alert_agent
from ai.alert_agent import RiskAssessment, _describe_static_objects, agent


ESTADO_BASE = {
    "frame": None,
    "static_objects": [{"track_id": 1, "class_name": "backpack", "center": (10, 20), "static_frames": 90}],
    "scene_description": "",
    "risk_level": "",
    "risk_reason": "",
    "risk_confidence": 0.0,
    "alert_message": "",
}


class AnalizadorFalso:
    """Doble del analizador de escena: no llama al modelo de visión."""

    def analyze(self, frame, context=""):
        return "Una mochila sola."


class RiskLLMFalso:
    """Doble del LLM de riesgo: devuelve siempre la evaluación indicada."""

    def __init__(self, evaluacion):
        self.evaluacion = evaluacion

    def invoke(self, _mensajes):
        return self.evaluacion


@pytest.fixture
def llm_falso(monkeypatch):
    """Sustituye las llamadas al LLM; devuelve un setter para fijar el riesgo evaluado.

    Se reemplazan los objetos completos en el módulo (y no sus métodos), porque `risk_llm`
    es un RunnableSequence de pydantic y no admite que se le asignen atributos.
    """
    monkeypatch.setattr(alert_agent, "analyzer", AnalizadorFalso())

    def fijar_riesgo(nivel, confianza=0.9):
        evaluacion = RiskAssessment(risk_level=nivel, reason="motivo de prueba", confidence=confianza)
        monkeypatch.setattr(alert_agent, "risk_llm", RiskLLMFalso(evaluacion))

    return fijar_riesgo


@pytest.mark.parametrize("nivel", ["medium", "high"])
def test_riesgo_relevante_genera_alerta(llm_falso, nivel):
    llm_falso(nivel)

    resultado = agent.invoke(dict(ESTADO_BASE))

    assert resultado["risk_level"] == nivel
    assert resultado["alert_message"]
    assert nivel in resultado["alert_message"]
    assert "motivo de prueba" in resultado["alert_message"]


def test_riesgo_bajo_no_genera_alerta(llm_falso):
    llm_falso("low")

    resultado = agent.invoke(dict(ESTADO_BASE))

    assert resultado["risk_level"] == "low"
    assert resultado["alert_message"] == ""


def test_la_alerta_incluye_la_confianza(llm_falso):
    llm_falso("high", confianza=0.75)

    resultado = agent.invoke(dict(ESTADO_BASE))

    assert "75%" in resultado["alert_message"]


def test_la_descripcion_de_escena_llega_al_estado(llm_falso):
    llm_falso("low")

    resultado = agent.invoke(dict(ESTADO_BASE))

    assert resultado["scene_description"] == "Una mochila sola."


def test_describe_objetos_estaticos_detalla_clase_y_tiempo():
    texto = _describe_static_objects(ESTADO_BASE["static_objects"])

    assert "1 objeto(s)" in texto
    assert "backpack" in texto
    assert "90 frames" in texto


def test_describe_objetos_estaticos_sin_objetos():
    assert _describe_static_objects([]) == "No se han detectado objetos estáticos."
