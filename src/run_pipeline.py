import cv2
import os
import queue
import threading
import time
from detection.yolo_detector import YOLODetector
from capture.video_source import VideoSource, VideoSourceError, EndOfStream
from detection.tracker import Tracker
from detection.event_detector import EventDetector
from ai.alert_agent import agent
from config import FRAMES_DIR, FRAME_SIZE, ANALYSIS_INTERVAL, REANALYSIS_STEP
from datetime import datetime

# Traza de diagnóstico opcional: imprime qué vio y qué decidió el LLM en cada análisis.
# Desactivada por defecto; se activa con la variable de entorno VIGIA_DEBUG (p. ej. VIGIA_DEBUG=1).
DEBUG = os.environ.get("VIGIA_DEBUG", "").lower() not in ("", "0", "false", "no")


def _format_video_time(seconds):
    """Formatea un instante del vídeo (en segundos) como mm:ss o hh:mm:ss."""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _analysis_worker(job_queue, events, video):
    """
    Consume en segundo plano los trabajos de análisis de escena encolados por el pipeline.

    Cada trabajo es una tupla (frame, annotated_frame, static_objects, video_time). Invoca al
    agente de IA (llamada lenta al LLM) sin bloquear el bucle de captura y, si el riesgo es
    medio/alto, guarda la captura y registra el evento. El análisis se hace sobre el frame
    **crudo**, mientras que la captura que se persiste es el **anotado**, para que en el evento
    se vea qué objeto lo disparó (caja e ID). `video_time` es el instante del vídeo (en segundos)
    en que se capturó el frame, o None en fuentes en directo. Se detiene al recibir el centinela None.

    Registra por consola los errores (con su tipo) y un resumen final, para poder distinguir
    «no había riesgo» de «las llamadas al LLM fallaron», que desde el dashboard se ven igual:
    cero eventos.
    """
    analizados = guardados = fallidos = 0
    errores = {}
    while True:
        job = job_queue.get()
        try:
            if job is None:
                break
            frame, annotated_frame, static_objects, video_time = job
            analizados += 1
            result = agent.invoke({
                "frame": frame,
                "static_objects": static_objects,
                "scene_description": "",
                "risk_level": "",
                "risk_reason": "",
                "risk_confidence": 0.0,
                "alert_message": ""
            })
            # Traza de diagnóstico (solo con VIGIA_DEBUG): qué vio y qué decidió el LLM.
            if DEBUG:
                clases = ", ".join(f"{o.get('class_name')}({o.get('static_frames')}f)" for o in static_objects)
                t = _format_video_time(video_time) if video_time is not None else "live"
                print(f"[analisis] t={t} riesgo={result['risk_level']} | estaticos=[{clases}] | motivo={result.get('risk_reason','')}")
            if events is not None and result["risk_level"] in ("medium", "high"):
                # Instante del vídeo (mm:ss) si conocemos los FPS; si no (webcam/stream), hora real.
                if video_time is not None:
                    timestamp = _format_video_time(video_time)
                else:
                    timestamp = datetime.now().isoformat()
                # Nombre de archivo único e independiente del timestamp mostrado al usuario.
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                for obj in static_objects:
                    filename = f"{obj['track_id']}_{stamp}.jpg"
                    frame_path = str(FRAMES_DIR / filename)
                    # Se guarda el frame anotado: muestra la caja y el ID del objeto detectado.
                    cv2.imwrite(frame_path, annotated_frame)
                    events.add_event(track_id=obj["track_id"], alert=result["alert_message"], risk_level=result["risk_level"], timestamp=timestamp, frame_path=frame_path, video=video)
                    guardados += 1
        except Exception as e:
            fallidos += 1
            tipo = type(e).__name__
            errores[tipo] = errores.get(tipo, 0) + 1
            print(f"[analisis] ERROR {tipo}: {e}")
        finally:
            job_queue.task_done()

    resumen = f"[analisis] RESUMEN: {analizados} analizados, {guardados} evento(s) guardado(s), {fallidos} fallido(s)"
    if errores:
        resumen += " | errores: " + ", ".join(f"{k}x{v}" for k, v in errores.items())
    print(resumen)


def run_pipeline(video_source=0, events=None, latest_frame=None, stop_event=None, video=None, progress=None, model_path="yolov8n.pt", confidence=0.5, show_window=False):
    """
    Función principal para ejecutar el pipeline de visión por computador.
    Esta función abre la fuente de video, carga el modelo YOLO y procesa cada frame para detectar objetos.

    El bucle termina cuando se activa `stop_event` (p. ej. desde el endpoint /stop) o cuando
    se acaba el vídeo. La ventana de OpenCV es opcional y solo tiene sentido en modo standalone.

    Args:
        video_source: Fuente de video (puede ser un índice de cámara o una ruta de archivo).
        events: EventStore donde registrar los eventos detectados (opcional).
        latest_frame: Lista de un elemento donde publicar el último frame anotado (opcional).
        stop_event: threading.Event para detener el bucle desde fuera (opcional).
        video: Identificador del vídeo al que asociar los eventos. Si es None, se deriva de video_source.
        progress: dict compartido donde publicar el avance (claves processed/total/percent) (opcional).
        model_path: Ruta al modelo YOLO a utilizar.
        confidence: Umbral de confianza para las detecciones.
        show_window: Si es True, muestra una ventana de OpenCV con el vídeo anotado (pulsa 'q' para salir).
                     Por defecto False: pensado para ejecución bajo la API, donde los frames se sirven
                     vía /latest_frame y /stream y el control se hace con stop_event.
    """
    if video is None:
        video = os.path.basename(str(video_source)) if isinstance(video_source, str) else str(video_source)

    detector = YOLODetector(model_path=model_path, confidence=confidence)
    tracker = Tracker(max_age=30)
    event_detector = EventDetector(static_threshold=30)
    
    last_analysis_time = 0
    analysis_interval = ANALYSIS_INTERVAL

    # El análisis de escena (LLM) corre en un hilo aparte para no bloquear la captura.
    # maxsize=1: si ya hay un análisis pendiente, se descarta el nuevo (lo limita el intervalo).
    job_queue = queue.Queue(maxsize=1)
    worker = threading.Thread(target=_analysis_worker, args=(job_queue, events, video), daemon=True)
    worker.start()

    try:
        with VideoSource(video_source) as source:
            total_frames = source.frame_count()
            fps = source.fps()
            processed_frames = 0
            encolados = descartados = 0
            frames_con_estaticos = 0
            # track_id -> nº de frames inmóvil en su último análisis. Sirve para no re-analizar
            # el mismo objeto salvo que persista bastante más tiempo (escalada de riesgo).
            analyzed_static = {}
            if progress is not None:
                progress.update({"processed": 0, "total": total_frames, "percent": 0.0})

            while stop_event is None or not stop_event.is_set():
                try:
                    frame = source.read()
                    processed_frames += 1
                    if progress is not None:
                        progress["processed"] = processed_frames
                        progress["percent"] = round(processed_frames / total_frames * 100, 1) if total_frames else 0.0
                    frame = cv2.resize(frame, FRAME_SIZE)
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections, frame)
                    annotated_frame = tracker.annotate(frame, tracks)
                    if latest_frame is not None:
                        latest_frame[0] = annotated_frame.copy()

                    if show_window:
                        cv2.imshow("Detections", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    static_objects = event_detector.update(tracks)
                    if static_objects:
                        frames_con_estaticos += 1
                        # Olvida los objetos que ya no están estáticos (así, si uno reaparece
                        # inmóvil más tarde, se vuelve a analizar como nuevo).
                        current_ids = {o["track_id"] for o in static_objects}
                        analyzed_static = {t: c for t, c in analyzed_static.items() if t in current_ids}
                        # Objetos que justifican un análisis: nuevos, o que llevan REANALYSIS_STEP
                        # frames más inmóviles desde la última vez (escalada de posible abandono).
                        nuevos = [
                            o for o in static_objects
                            if o["track_id"] not in analyzed_static
                            or o["static_frames"] - analyzed_static[o["track_id"]] >= REANALYSIS_STEP
                        ]
                        now = time.time()
                        if nuevos and now - last_analysis_time > analysis_interval:
                            # Segundo del vídeo correspondiente a este frame (None en directo).
                            video_time = processed_frames / fps if fps else None
                            try:
                                # Se encolan los dos frames: el crudo para el LLM (las cajas
                                # dibujadas podrían condicionar su análisis) y el anotado para
                                # guardarlo como captura del evento.
                                job_queue.put_nowait(
                                    (frame.copy(), annotated_frame.copy(), static_objects, video_time)
                                )
                                last_analysis_time = now
                                encolados += 1
                                # Registra el nº de frames inmóvil con que se analizó cada uno.
                                for o in nuevos:
                                    analyzed_static[o["track_id"]] = o["static_frames"]
                            except queue.Full:
                                # ya hay un análisis en curso; se omite este
                                descartados += 1

                except EndOfStream:
                    # Fin normal del vídeo: no es un error.
                    print(f"Procesamiento finalizado: {video}")
                    print(
                        f"[vision] {processed_frames} frames, {frames_con_estaticos} con objetos estaticos, "
                        f"{encolados} analisis encolados, {descartados} descartados (LLM ocupado)"
                    )
                    if progress is not None and total_frames:
                        progress.update({"processed": total_frames, "percent": 100.0})
                    break
                except VideoSourceError as e:
                    print(f"Error al procesar el frame: {e}")
                    break
            if show_window:
                cv2.destroyAllWindows()
    finally:
        job_queue.put(None)   # centinela: detiene el worker tras vaciar lo pendiente
        worker.join()

if __name__ == "__main__":
    run_pipeline(video_source=0, confidence=0.3, show_window=True)