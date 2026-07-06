"""
Analizador de escenas para procesar frames de video y generar descripciones.
Este módulo define la clase SceneAnalyzer, que utiliza un modelo de lenguaje para analizar el contenido de los frames de video y generar descripciones textuales de lo que se observa en la escena.
"""

import base64
import cv2
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from config import GROQ_MODEL

class SceneAnalyzer:
    """
    Clase para analizar escenas utilizando un modelo de lenguaje.
    Esta clase utiliza un modelo de lenguaje para analizar el contenido de los frames de video y 
    generar descripciones textuales de lo que se observa en la escena.
    """
    
    def __init__(self):
        """
        Inicializa el analizador de escenas y carga las variables de entorno necesarias.
        """
        load_dotenv()
        self.analyzer = ChatGroq(model=GROQ_MODEL)
        
    def analyze(self, frame, context=""):
        """
        Analiza un frame de video y genera una descripción textual de la escena.
        """
        _, buffer = cv2.imencode(".jpg", frame)
        image_b64 = base64.b64encode(buffer).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "text", "text": "¿Qué ves en esta imagen? " + context},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ])

        response = self.analyzer.invoke([message])
        return response.content