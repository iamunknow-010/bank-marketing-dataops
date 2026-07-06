import pandas as pd
from src.utils.logger import logger
from src.pipeline.validation.schemas import EXPECTED_SCHEMA

def validar_estructura(df: pd.DataFrame) -> bool:
    logger.info("--- Iniciando Validación Estructural de Esquema y Tipos ---")
    
    missing_cols = [col for col in EXPECTED_SCHEMA if col not in df.columns]
    if missing_cols:
        logger.error(f"Falla Estructural: Faltan columnas requeridas: {missing_cols}")
        return False
        
    for col, expected_type in EXPECTED_SCHEMA.items():
        if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Falla de Tipo: '{col}' debería ser numérico, pero llegó como {df[col].dtype}")
            return False
        elif expected_type == 'object' and not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            logger.error(f"Falla de Tipo: '{col}' debería ser texto/categoría, pero llegó como {df[col].dtype}")
            return False
    logger.info("Validación estructural superada: Nombres y tipos de datos coherentes.")
    return True