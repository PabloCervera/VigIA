"""
Interfaz de usuario en Streamlit para controlar el pipeline de visión y explorar sus resultados.
Se comunica con la API (por defecto en http://localhost:8000) y organiza la aplicación en tres
fases: subir un vídeo (idle), procesarlo (processing) y revisar los eventos con un chat de
preguntas sobre la escena (finished).
"""

import os
import streamlit as st
import requests

# URL base de la API; configurable por entorno para despliegues (p. ej. Docker Compose).
API = os.environ.get("API_URL", "http://localhost:8000")

def dashboard():
    """Renderiza la aplicación y enruta entre las fases idle / processing / finished."""
    st.title("Dashboard de Eventos")

    if "phase" not in st.session_state:
        st.session_state.phase = "idle"
    
    if st.session_state.phase == "idle":
        st.subheader("Sube un video para iniciar el análisis")
        upload_file()

    elif st.session_state.phase == "processing":
        st.subheader("Procesando vídeo...")
        show_progress()
        check_status()
        if st.button("Detener"):
            requests.post(f"{API}/stop")
            st.session_state.phase = "finished"
            st.rerun()

    elif st.session_state.phase == "finished":
        st.subheader("Eventos detectados")
        show_results()

        if st.button("Volver"):
            requests.post(f"{API}/clear_events")
            # El historial del chat se refiere a los eventos que se acaban de borrar:
            # hay que descartarlo también para empezar limpio con el siguiente vídeo.
            st.session_state.messages = []
            st.session_state.phase = "idle"
            st.rerun()

        st.subheader("Pregunta sobre la escena")
        chat_qa()

def upload_file():
    """Permite subir un vídeo, lo envía a la API e inicia el procesamiento."""
    uploaded_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        if st.button("Iniciar"):
            st.write("Subiendo y procesando el video...")
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{API}/upload_video", files=files)
            if response.status_code == 200:
                video_path = response.json()["video_path"]
                requests.post(f"{API}/start", json={"path": video_path})
                st.session_state.phase = "processing"
                st.rerun()
            else:
                st.error("Error al subir el video.")       

@st.fragment(run_every=5)
def show_progress():
    """Muestra el avance del procesamiento del vídeo actual; indeterminado si es un flujo en directo."""
    response = requests.get(f"{API}/progress")
    if response.status_code == 200:
        data = response.json()
        if data["total"]:
            st.progress(data["percent"] / 100, text=f"{data['percent']:.0f}% ({data['processed']}/{data['total']} frames)")
        else:
            st.write("Procesando flujo en directo...")

@st.fragment(run_every=5)
def check_status():
    """Sondea periódicamente el estado de la API y pasa a la fase final cuando el pipeline se detiene."""
    response = requests.get(f"{API}/status")
    if response.status_code == 200:
        status = response.json()
        if status['status'] == "stopped":
            st.session_state.phase = "finished"
            st.rerun()

def chat_qa():
    """Muestra el chat y envía cada pregunta del usuario al endpoint `/ask` de la API."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Escribe tu pregunta...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        response = requests.post(f"{API}/ask", json={"text": question})
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

def show_results():
    """Recupera los eventos de la API y los muestra con su nivel de riesgo, alerta y captura."""
    response = requests.get(f"{API}/events")
    if response.status_code == 200:
        events = response.json()["events"]
        riesgo_es = {"low": "bajo", "medium": "medio", "high": "alto"}
        st.write(f"Total de eventos: {len(events)}")
        for event in events:
            riesgo = riesgo_es.get(event['risk_level'], event['risk_level'])
            st.write(f"**Minuto del vídeo {event['timestamp']}** — Riesgo: {riesgo}")
            st.write(event['alert'])
            if event['frame_path']:
                st.image(event['frame_path'])
            st.divider()
        
@st.fragment(run_every=0.1)
def update_frame():
    """Refresca de forma continua el último frame anotado obtenido de la API."""
    response = requests.get(f"{API}/latest_frame")
    if response.status_code == 200:
        st.image(response.content, channels="BGR")
        
@st.fragment(run_every=1)
def update_events():
    """Refresca periódicamente la lista de eventos obtenida de la API."""
    response = requests.get(f"{API}/events")
    if response.status_code == 200:
        events = response.json()
        st.write(f"Total de eventos: {len(events['events'])}")
        for event in events['events']:
            st.write(event)

if __name__ == "__main__":
    dashboard()