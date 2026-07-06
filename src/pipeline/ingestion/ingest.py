import pandas as pd
from src.config import RUTA_DATOS_RAW
from src.utils.logger import logger

def ejecutar_ingesta() -> pd.DataFrame:
    logger.info("Iniciando Ingesta (Capa Bronce): Extracción a memoria...")
    
    ruta_archivo = RUTA_DATOS_RAW / "02_bank.csv"
    
    if not ruta_archivo.exists():
        logger.error(f"Fallo crítico: Archivo origen no encontrado en {ruta_archivo}")
        raise FileNotFoundError(f"Falta el archivo requerido: {ruta_archivo}")
        
    try:
        # Cargamos el DataFrame una sola vez y lo mantenemos en memoria
        df = pd.read_csv(ruta_archivo)
        
        # Validaciones estructurales rápidas
        if df.empty:
            raise ValueError("El archivo CSV está vacío.")
            
        logger.info(f"Extracción exitosa. Volumen: {df.shape[0]} filas | {df.shape[1]} columnas.")
        return df
        
    except Exception as e:
        logger.error(f"Fallo en I/O durante ingesta: {str(e)}")
        raise e