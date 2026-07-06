# VigIA

> Un **vig**ía con **IA**: vigilancia de vídeo que detecta y sigue objetos en tiempo real y delega el análisis de riesgo en un agente de IA generativa.

Pipeline de **visión por computador** que detecta y sigue objetos en tiempo real sobre un flujo de vídeo (webcam, fichero o RTSP) y, cuando identifica situaciones potencialmente relevantes (objetos que quedan estáticos), delega en un **agente de IA generativa** que describe la escena, evalúa el nivel de riesgo y emite una alerta. Los eventos se persisten en una base de datos y se exploran desde un **dashboard web** que incluye un **chat para hacer preguntas** sobre lo ocurrido.

Combina:

- **YOLOv8** ([ultralytics](https://github.com/ultralytics/ultralytics)) para detección de objetos.
- **DeepSORT** ([deep-sort-realtime](https://github.com/levan92/deep_sort_realtime)) para seguimiento multi-objeto con IDs persistentes.
- **LangChain + LangGraph** orquestando un agente con el modelo de visión `llama-4-scout` de **Groq** para analizar la escena y razonar sobre el riesgo.
- **FastAPI** para controlar el pipeline y exponer eventos vía API REST + WebSocket.
- **SQLite** para almacenar el historial de eventos.
- **Streamlit** como interfaz: subir vídeo, ver alertas con su captura y preguntar sobre la escena.

> ⚠️ El proyecto está en desarrollo activo (organizado por sprints). Ver [Estado del proyecto](#estado-del-proyecto).

---

## Arquitectura

```
┌──────────────┐  frames  ┌──────────────┐ detecciones ┌──────────┐ tracks ┌────────────────┐
│ VideoSource  │────────▶ │ YOLODetector │───────────▶ │ Tracker  │──────▶ │ EventDetector  │
│ (cam/file/   │          │ (YOLOv8)     │             │(DeepSORT)│        │ (objetos       │
│  RTSP)       │          └──────────────┘             └──────────┘        │  estáticos)    │
└──────────────┘                                                           └───────┬────────┘
                                                                                   │ static_objects
                                                                                   ▼
                                                   ┌─────────────────────────────────────────┐
                                                   │  Agente de alerta (LangGraph)             │
                                                   │  analyze_scene → decide_risk → ┬─ alta/med│
                                                   │   (SceneAnalyzer/Groq Vision)  └─ baja ─┐ │
                                                   │                          send_alert / ignore
                                                   └─────────────────┬─────────────────────────┘
                                                                     │ alert + frame (si riesgo medio/alto)
                                                                     ▼
                                       ┌──────────────────┐    ┌─────────────────────┐
                                       │  EventStore      │◀──▶│  FastAPI (api.py)   │
                                       │  (SQLite)        │    │  REST + WebSocket   │
                                       └──────────────────┘    └──────────┬──────────┘
                                                  ▲                       │
                                          ┌───────┴────────┐              ▼
                                          │  QAChain (Groq)│     ┌────────────────────┐
                                          │  chat sobre    │     │ Dashboard           │
                                          │  los eventos   │     │ (Streamlit)         │
                                          └────────────────┘     └────────────────────┘
```

### Flujo del pipeline ([src/run_pipeline.py](src/run_pipeline.py))

1. Se abre la fuente de vídeo y se lee frame a frame.
2. Cada frame se redimensiona y pasa por **YOLODetector** → lista de `Detection`.
3. **Tracker** (DeepSORT) asigna un ID estable a cada objeto entre frames y anota el frame.
4. **EventDetector** mantiene el historial de posiciones de cada track y detecta los que llevan estáticos un nº de frames (posible objeto abandonado, persona inmóvil, etc.).
5. Si hay objetos estáticos —y respetando un intervalo mínimo entre análisis (`analysis_interval = 10s`)— se **encola** un trabajo de análisis. La llamada al LLM corre en un **hilo aparte** para no bloquear la captura, de modo que el vídeo se sigue procesando con fluidez.
6. El agente describe la escena, decide el riesgo y, si es **medio o alto**, guarda la captura en disco y registra el evento en la base de datos.

### El agente de alerta ([src/ai/alert_agent.py](src/ai/alert_agent.py))

Implementado como un grafo de estados con **LangGraph**:

| Nodo            | Función                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| `analyze_scene` | Envía el frame al modelo de visión de Groq y obtiene una descripción textual. |
| `decide_risk`   | Pide al LLM que clasifique el riesgo en `low` / `medium` / `high`.          |
| `send_alert`    | (riesgo medio/alto) Compone el mensaje de alerta.                           |
| `ignore`        | (riesgo bajo) No hace nada.                                                 |

El enrutado entre `send_alert` e `ignore` se decide mediante una **arista condicional** según el `risk_level`.

### Consulta sobre los eventos ([src/ai/qa_chain.py](src/ai/qa_chain.py))

`QAChain` recibe la pregunta del usuario junto con los eventos recientes de la base de datos y pide al LLM una respuesta basada **únicamente** en ese historial. Es lo que alimenta el chat del dashboard.

---

## Estructura del repositorio

```
vigia/
├── src/
│   ├── capture/
│   │   └── video_source.py      # Abstracción de fuente de vídeo (webcam/fichero/RTSP)
│   ├── detection/
│   │   ├── detection.py         # Dataclass Detection (bbox, center, area)
│   │   ├── yolo_detector.py     # Wrapper de YOLOv8
│   │   ├── tracker.py           # Tracking con DeepSORT + anotación
│   │   └── event_detector.py    # Detección de objetos estáticos
│   ├── database/
│   │   └── event_store.py       # Persistencia de eventos en SQLite
│   ├── ai/
│   │   ├── scene_analyzer.py    # Descripción de escena (Groq Vision)
│   │   ├── alert_agent.py       # Agente LangGraph de análisis de riesgo
│   │   └── qa_chain.py          # Chat Q&A sobre los eventos detectados
│   ├── config.py                # Rutas centralizadas (data/, BD, capturas)
│   ├── run_pipeline.py          # Bucle principal del pipeline
│   ├── api.py                   # API FastAPI + WebSocket
│   └── utils.py                 # Paleta de colores para anotaciones
├── dashboard.py                 # Interfaz Streamlit
├── data/                        # Datos generados en ejecución (no versionado)
├── tests/                       # Tests unitarios (pytest)
├── pytest.ini
├── Dockerfile                   # Imagen del proyecto
├── docker-compose.yml           # Orquestación de API + dashboard
├── .env.example                 # Plantilla de variables de entorno
├── requirements.txt
└── README.md
```

---

## Requisitos

- **Python 3.10+**
- Una **API key de Groq** (para `scene_analyzer`, `alert_agent` y `qa_chain`).
- Pesos de YOLOv8 (`yolov8n.pt`) — `ultralytics` los descarga automáticamente la primera vez.

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd vigia

# 2. Crear y activar un entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # Linux/macOS

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Configuración

Crea un fichero `.env` en la raíz del proyecto con tu clave de Groq:

```env
GROQ_API_KEY=tu_api_key_aqui
```

`load_dotenv()` la carga automáticamente en los módulos que usan Groq.

---

## Uso

La forma recomendada de usar el proyecto es levantar la API y el dashboard, que se comunican entre sí.

### 1. Arrancar la API

```bash
cd src
uvicorn api:app --reload
```

### 2. Arrancar el dashboard (en otra terminal)

```bash
streamlit run dashboard.py
```

Desde el dashboard puedes **subir un vídeo**, **iniciar y detener** el procesamiento o esperar a que termine, ver en tiempo real las **alertas detectadas con su captura** y, al terminar, **hacer preguntas en lenguaje natural** sobre lo ocurrido en la escena. El dashboard asume que la API corre en `http://localhost:8000`.

### Endpoints de la API

| Endpoint         | Método    | Descripción                                                       |
| ---------------- | --------- | ----------------------------------------------------------------- |
| `/upload_video`  | POST      | Sube un vídeo al servidor y devuelve su ruta.                     |
| `/start`         | POST      | Inicia el pipeline (en un hilo) sobre la ruta indicada.          |
| `/stop`          | POST      | Detiene el pipeline en ejecución.                                |
| `/status`        | GET       | Estado del pipeline (`running`/`stopped`) y nº total de eventos. |
| `/progress`      | GET       | Avance del procesamiento (frames procesados, total y porcentaje).|
| `/events`        | GET       | Eventos del vídeo actual.                                        |
| `/clear_events`  | POST      | Elimina los eventos del vídeo actual.                            |
| `/ask`           | POST      | Pregunta sobre los eventos del vídeo actual; responde vía `QAChain`. |
| `/latest_frame`  | GET       | Último frame anotado (JPEG).                                     |
| `/stream`        | WebSocket | Streaming de frames anotados en tiempo real.                     |

### Tests

```bash
pytest
```

Tests unitarios de la lógica de dominio (detección de objetos estáticos y persistencia de eventos), sin dependencias externas (usan SQLite en memoria y stubs de tracks).

---

## Ejecución con Docker

Levanta la API y el dashboard como dos servicios con un solo comando:

```bash
# 1. Configura tu clave de Groq
cp .env.example .env        # y edita GROQ_API_KEY

# 2. Construye y arranca
docker compose up --build
```

- Dashboard: [http://localhost:8501](http://localhost:8501)
- API: [http://localhost:8000](http://localhost:8000)

Ambos servicios comparten un volumen `app-data` donde persisten los vídeos subidos, las capturas y la base de datos. El dashboard localiza la API mediante la variable `API_URL` (por defecto `http://api:8000` dentro de Compose; `http://localhost:8000` en local).

---

## Parámetros principales

| Componente       | Parámetro          | Por defecto | Significado                                                      |
| ---------------- | ------------------ | ----------- | --------------------------------------------------------------- |
| `YOLODetector`   | `confidence`       | `0.5`       | Umbral mínimo de confianza para una detección.                  |
| `Tracker`        | `max_age`          | `30`        | Frames que un track sobrevive sin detecciones.                  |
| `EventDetector`  | `static_threshold` | `30`        | Nº de frames de historial para evaluar si un objeto está quieto.|
| `EventDetector`  | `max_distance`     | `10`        | Distancia (px) por debajo de la cual se considera estático.     |
| `run_pipeline`   | `analysis_interval`| `10` (s)    | Tiempo mínimo entre invocaciones al agente de IA.               |

---

## Estado del proyecto

El desarrollo está organizado por sprints (ver historial de commits):

- ✅ **Sprint 1–2** — Captura de vídeo, detección YOLOv8, tracking DeepSORT, detección de objetos estáticos y persistencia de eventos en SQLite.
- ✅ **Sprint 3** — Descripción de escena con Groq Vision y agente de riesgo con LangGraph.
- ✅ **Sprint 4** — API FastAPI (control del pipeline, WebSocket de streaming), dashboard de Streamlit y chat de Q&A sobre los eventos.

### Tareas pendientes

- [ ] Integrar un bot de Telegram para notificaciones.
- [ ] Ampliar la cobertura de tests (lógica de riesgo del agente con mocks).

---

## Stack tecnológico

| Área                  | Tecnología                                              |
| --------------------- | ------------------------------------------------------- |
| Visión por computador | OpenCV, YOLOv8 (ultralytics), DeepSORT                  |
| IA generativa         | LangChain, LangGraph, Groq (`llama-4-scout-17b`)        |
| API / backend         | FastAPI, Uvicorn, WebSockets                            |
| Persistencia          | SQLite                                                  |
| Dashboard             | Streamlit                                               |
| Despliegue            | Docker, Docker Compose                                  |
| Notificaciones (prev.)| python-telegram-bot                                     |
| Utilidades            | python-dotenv, pydantic, pillow, numpy                  |
| Testing               | pytest, pytest-asyncio                                  |
