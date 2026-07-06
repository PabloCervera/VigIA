"""
Este módulo define un agente de alerta que analiza la escena, evalúa el nivel de riesgo y genera alertas en función de la información obtenida. Utiliza un modelo de lenguaje para describir la escena y determinar el nivel de riesgo,
 y luego decide si enviar una alerta o ignorar la situación.
"""

from typing import Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from ai.scene_analyzer import SceneAnalyzer
from config import GROQ_MODEL

class AgentState(TypedDict):
    frame: object
    static_objects: list
    scene_description: str
    risk_level: str
    risk_reason: str
    risk_confidence: float
    alert_message: str


class RiskAssessment(BaseModel):
    """Evaluación de riesgo estructurada que debe producir el LLM."""
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Nivel de riesgo de la escena según la política indicada."
    )
    reason: str = Field(
        description="Justificación breve (1-2 frases) basada en la escena y los objetos estáticos."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confianza de 0 a 1 en la evaluación."
    )


# Política de riesgo (rúbrica) fundamentada en las señales de las que disponemos:
# tipo de objeto, tiempo inmóvil y número de objetos estáticos.
RISK_RUBRIC = """Eres un sistema de videovigilancia que clasifica el riesgo de una escena.
Aplica esta política:

- high: objeto portátil típico de abandono (mochila, maleta, bolso, caja, paquete) inmóvil
        durante mucho tiempo y sin una persona junto a él; o una persona completamente inmóvil
        durante un periodo prolongado (posible desvanecimiento o incidente).
- medium: objeto inmóvil cuya naturaleza no queda clara, o persona detenida de forma prolongada;
          situación ambigua que conviene que un operador revise.
- low: elementos fijos por naturaleza (mobiliario, vehículos estacionados, plantas, señales),
       sombras o reflejos, peatones detenidos brevemente, o escena sin nada preocupante.

Cuanto más tiempo lleve inmóvil un objeto sospechoso, mayor el riesgo. Sé conservador:
ante poca evidencia, no escales el nivel."""


analyzer = SceneAnalyzer()
load_dotenv()
llm = ChatGroq(model=GROQ_MODEL)
risk_llm = llm.with_structured_output(RiskAssessment)


def _describe_static_objects(static_objects: list) -> str:
    """Construye una descripción textual de los objetos estáticos para fundamentar el análisis."""
    if not static_objects:
        return "No se han detectado objetos estáticos."
    lineas = [
        f"- {obj.get('class_name', 'objeto')} (id {obj['track_id']}) inmóvil durante "
        f"{obj.get('static_frames', '?')} frames, centrado en {obj['center']}."
        for obj in static_objects
    ]
    return f"Se han detectado {len(static_objects)} objeto(s) estático(s):\n" + "\n".join(lineas)


def analyze_scene(state: AgentState) -> dict:
    """
    Analiza la escena con el modelo de visión para generar una descripción textual,
    aportándole como contexto los objetos estáticos detectados.
    """
    context = _describe_static_objects(state["static_objects"])
    description = analyzer.analyze(state["frame"], context)
    return {"scene_description": description}

def decide_risk(state: AgentState) -> dict:
    """
    Evalúa el riesgo combinando la descripción de la escena con los datos de los objetos
    estáticos (clase y tiempo inmóvil), y devuelve una evaluación estructurada.
    """
    prompt = (
        f"{RISK_RUBRIC}\n\n"
        f"Descripción de la escena:\n{state['scene_description']}\n\n"
        f"Objetos estáticos detectados:\n{_describe_static_objects(state['static_objects'])}\n\n"
        f"Evalúa el nivel de riesgo siguiendo la política."
    )
    assessment = risk_llm.invoke([HumanMessage(content=prompt)])
    return {
        "risk_level": assessment.risk_level,
        "risk_reason": assessment.reason,
        "risk_confidence": assessment.confidence,
    }

def send_alert(state: AgentState) -> dict:
    """
    Genera un mensaje de alerta con el nivel de riesgo, su confianza y la justificación.
    """
    alert_message = (
        f"Alerta nivel {state['risk_level']} "
        f"(confianza {state['risk_confidence']:.0%}): {state['risk_reason']}"
    )
    return {"alert_message": alert_message}

def ignore(state: AgentState) -> dict:
    """
    Acción de ignorar, no realiza ninguna operación.
    """
    return {}


graph = StateGraph(AgentState)

# Añadir nodos
graph.add_node("analyze_scene", analyze_scene)
graph.add_node("decide_risk", decide_risk)
graph.add_node("send_alert", send_alert)
graph.add_node("ignore", ignore)

# Punto de entrada
graph.set_entry_point("analyze_scene")

# Aristas fijas
graph.add_edge("analyze_scene", "decide_risk")

# Arista condicional: según risk_level va a send_alert o ignore
graph.add_conditional_edges(
    "decide_risk",
    lambda state: state["risk_level"] if state["risk_level"] in ["high", "medium"] else "low",
    {"high": "send_alert", "medium": "send_alert", "low": "ignore"}
)

graph.add_edge("send_alert", END)
graph.add_edge("ignore", END)

agent = graph.compile()