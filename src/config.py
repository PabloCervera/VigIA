"""
Rutas centralizadas del proyecto.
Resuelve la ubicación de los datos generados en ejecución de forma relativa a la raíz
del proyecto (y no al directorio de trabajo), de modo que el pipeline, la API y la base
de datos escriban siempre en el mismo sitio independientemente de desde dónde se lancen.
"""

import os
from pathlib import Path

# Raíz del proyecto (carpeta que contiene a src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Datos generados en ejecución, todos bajo data/.
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"   # vídeos subidos por el usuario
FRAMES_DIR = DATA_DIR / "frames"     # capturas de los eventos detectados
DB_PATH = DATA_DIR / "events.db"     # base de datos SQLite de eventos

# Garantiza que los directorios existan antes de escribir en ellos.
for _dir in (UPLOADS_DIR, FRAMES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Modelo de Groq usado por los módulos de IA (configurable por entorno).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")

# Parámetros del pipeline.
FRAME_SIZE = (1708, 960)   # tamaño (ancho, alto) al que se redimensiona cada frame
ANALYSIS_INTERVAL = 10     # segundos mínimos entre análisis del agente de IA
# Un objeto ya analizado no se vuelve a analizar hasta que acumule este nº de frames
# adicionales inmóvil (escalada de riesgo). Evita re-analizar la misma escena estática una
# y otra vez, pero permite reevaluar un objeto que persiste (posible abandono confirmado).
REANALYSIS_STEP = 150
