"""
Este módulo contiene la clase QAChain, que se encarga de responder preguntas basadas en eventos detectados por un sistema de vigilancia inteligente.
Utiliza un modelo de lenguaje para procesar la pregunta del usuario y generar una respuesta basada en
los eventos proporcionados.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_MODEL

class QAChain:
    """
    Clase para manejar la lógica de preguntas y respuestas basadas en eventos detectados.
    Esta clase utiliza un modelo de lenguaje para procesar la pregunta del usuario y generar una respuesta basada en los eventos proporcionados.
    """
    def __init__(self):
        self.qaChain = ChatGroq(model=GROQ_MODEL)

    def ask(self, question, events):
        """
        Responde a una pregunta basada en los eventos detectados.
        """
        formated_events = "\n".join([f"ID: {event['track_id']}, Alerta: {event['alert']}, Nivel de riesgo: {event['risk_level']}, Timestamp: {event['timestamp']}" for event in events])
                
        prompt = f"""Eres un asistente de un sistema de vigilancia inteligente. 
            Responde a la pregunta del usuario basándote ÚNICAMENTE en los siguientes eventos detectados.
            Si no hay información suficiente para responder, dilo claramente.

            Eventos:
            {formated_events}

            Pregunta: {question}
            """
        
        message = HumanMessage(content=prompt)
        response = self.qaChain.invoke([message])
        return response.content