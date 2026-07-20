"""Configuración común a todos los tests.

Los módulos de IA construyen un cliente de Groq al importarse, y ese cliente exige que
exista una API key aunque no se llegue a usar. Se define aquí una clave ficticia para que
los tests puedan importarlos sin credenciales reales (p. ej. en CI). Ningún test hace
llamadas de red: las dependencias que hablan con Groq se sustituyen por dobles de prueba.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "clave-ficticia-para-tests")
