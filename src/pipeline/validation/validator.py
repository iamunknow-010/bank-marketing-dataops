import pandas as pd
from src.config import RUTA_DATOS_PROCESSED
from src.utils.logger import logger
from src.pipeline.validation.structural import validar_estructura
from src.pipeline.validation.semantic import validar_semantica

def ejecutar_validacion(df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=============================================================")
    logger.info("Iniciando validación estructural y semántica de los datos...")
    logger.info("=============================================================")

    if df is None:
        ruta_plata = RUTA_DATOS_PROCESSED / "bank_processed.csv"
        if not ruta_plata.exists():
            error_msg = f"Fallo de flujo: No se encontró el archivo procesado en {ruta_plata}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        df = pd.read_csv(ruta_plata)

    estructura_ok = validar_estructura(df)
    if not estructura_ok:
        logger.error("La validación estructural falló. Deteniendo proceso.")
        raise ValueError("Fallo crítico en la validación estructural del esquema.")
    
    mascara_validos = validar_semantica(df)
    
    df_validos = df[mascara_validos].copy()
    df_invalidos = df[~mascara_validos].copy()
    
    df_validos.to_csv(RUTA_DATOS_PROCESSED / "bank_final.csv", index=False)
    if not df_invalidos.empty:
        df_invalidos.to_csv(RUTA_DATOS_PROCESSED / "cuarentena.csv", index=False)
    
    logger.info("Proceso de validación finalizado.")
    logger.info(f" -> Registros Totales Evaluados: {len(df)}")
    logger.info(f" -> Registros VÁLIDOS (pasan al modelo): {len(df_validos)}")
    logger.info(f" -> Registros INVÁLIDOS (enviados a cuarentena): {len(df_invalidos)}")
    
    return df_validos, df_invalidos