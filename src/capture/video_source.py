"""
Módulo para manejar la fuente de video (cámara o archivo).
Este módulo define la clase VideoSource, que encapsula la lógica de abrir y cerrar la fuente de video.
También define la excepción VideoSourceError para manejar errores relacionados con la fuente de video.
"""

import cv2

class VideoSourceError(Exception):
    """Excepción personalizada para errores relacionados con la fuente de video."""
    pass

class EndOfStream(VideoSourceError):
    """Señala el fin normal de la fuente de vídeo (no quedan más frames)."""
    pass

class VideoSource:
    """
    Clase para manejar la fuente de video (cámara o archivo).
    Esta clase encapsula la lógica de abrir y cerrar la fuente de video, y proporciona una interfaz sencilla para acceder a los frames.
    """
    def __init__(self, source=0):
        """
        Inicializa la fuente de video.
        """
        self.source = source
        self._cap = None  # aún no abrimos nada

    def open(self):
        """
        Abre la fuente de video.
        Si ya está abierta, la cierra primero.
        
        Returns:
            VideoSource: la propia instancia, para poder encadenar llamadas.

        Raises:
            VideoSourceError: si no se puede abrir la fuente.
        """
        
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise VideoSourceError(f"No se pudo abrir: {self.source}")
        return self

    def close(self):
        """Libera los recursos de la fuente de vídeo."""
        if self._cap:
            self._cap.release()
            
    def read(self): 
        """
        Lee un frame de la fuente de video.
        
        Returns:
            Tuple[bool, np.ndarray]: Un booleano que indica si se leyó correctamente y el frame leído (en formato BGR).

        Raises:
            VideoSourceError: si la fuente no está abierta.
            EndOfStream: si no quedan más frames (fin del vídeo).
        """
        if not self._cap or not self._cap.isOpened():
            raise VideoSourceError("La fuente de video no está abierta.")
        ret, frame = self._cap.read()
        if not ret:
            raise EndOfStream("Fin del vídeo: no quedan más frames.")
        return frame

    def frame_count(self):
        """
        Devuelve el número total de frames del vídeo.

        Returns:
            int: Total de frames, o 0 si no se conoce (p. ej. webcam o stream RTSP).
        """
        if not self._cap:
            return 0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total if total > 0 else 0

    def fps(self):
        """
        Devuelve los fotogramas por segundo (FPS) del vídeo.

        Returns:
            float: FPS del vídeo, o 0.0 si se desconoce (p. ej. webcam o stream en directo).
        """
        if not self._cap:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 0.0

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()