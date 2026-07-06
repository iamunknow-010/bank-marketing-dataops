import os
import pandas as pd
from sqlalchemy import create_engine
from src.config import RUTA_DATOS_PROCESSED
from src.utils.logger import logger
from src.utils.security import anonimizar_datos

def ejecutar_carga(df: pd.DataFrame = None):
    logger.info("Iniciando etapa de Carga (Capa Oro) hacia Supabase...")
    
    # 1. Mantener la eficiencia: Usar memoria RAM si el flujo viene del main
    if df is None:
        ruta_entrada = RUTA_DATOS_PROCESSED / "bank_final.csv"
        if not ruta_entrada.exists():
            logger.error(f"Fallo de flujo: No se encontró el archivo validado en: {ruta_entrada}")
            raise FileNotFoundError(f"Falta bank_final.csv")
        df = pd.read_csv(ruta_entrada)
        
    try:
        # 2. Seguridad: Captura de credenciales dinámicas (Cumple con evitar hardcoding de datos sensibles)
        db_url = os.getenv("SUPABASE_DB_URL")
        if not db_url:
            raise ValueError("Falla de Seguridad: La variable SUPABASE_DB_URL no está configurada en el entorno.")
        
        if "ñ" in db_url:
            db_url = db_url.replace("ñ", "%C3%B1")
            
        # 3. Conexión a la infraestructura Cloud
        engine = create_engine(db_url)
        
        df_para_nube = anonimizar_datos(df)
        
        # 4. Inyección dinámica de la carga segura
        # ATENCIÓN: Ahora usamos df_para_nube, no df
        volumen_real = len(df_para_nube)
        logger.info(f"Inyectando {volumen_real} registros ENCRIPTADOS en Supabase... (Esto puede tardar unos segundos)")
        
        # El .to_sql ahora envía el dataframe anonimizado
        df_para_nube.to_sql("bank_marketing_gold", con=engine, if_exists="replace", index=False)
        
        logger.info("Datos cargados exitosamente en la base de datos destino con protección de seguridad.")
        
    except Exception as e:
        logger.error(f"Error crítico en la fase de carga a la nube: {str(e)}")
        raise e