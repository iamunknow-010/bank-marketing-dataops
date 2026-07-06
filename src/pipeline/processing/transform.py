import pandas as pd
from src.config import RUTA_DATOS_PROCESSED
from src.utils.logger import logger

def ejecutar_limpieza(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando Limpieza y Transformación (Capa Plata)...")
    
    try:
        # 1. Control de duplicados absolutos
        filas_iniciales = len(df)
        df = df.drop_duplicates().copy() # .copy() para evitar SettingWithCopyWarning
        logger.info(f"Limpieza de ruido: {filas_iniciales - len(df)} duplicados eliminados.")
        
        # 2. Imputación de valores nulos encubiertos ('unknown')
        # La rúbrica exige imputación. Usaremos la moda (el valor más frecuente) 
        # para no perder registros vitales ni alterar drásticamente la distribución.
        cols_con_unknown = ['job', 'education', 'contact', 'poutcome']
        for col in cols_con_unknown:
            if col in df.columns:
                # Calculamos la moda ignorando los 'unknown'
                moda = df[df[col] != 'unknown'][col].mode()[0]
                df[col] = df[col].replace('unknown', moda)
        logger.info(f"Imputación estadística aplicada sobre variables con 'unknown': {cols_con_unknown}")
        
        # 3. Transformación de variables binarias al formato matemático requerido (1/0)
        columnas_binarias = ['default', 'housing', 'loan', 'deposit']
        for col in columnas_binarias:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0})
        logger.info("Variables binarias codificadas exitosamente.")

        # 4. Optimización de Memoria (Reducción de carga computacional)
        # Los int64 consumen demasiada RAM sin necesidad. Hacemos downcast a enteros más pequeños.
        cols_enteras = df.select_dtypes(include=['int64']).columns
        for col in cols_enteras:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        logger.info("Tipos de datos optimizados (Downcasting int64 -> int16/32).")
        
        # 5. Guardar el estado limpio (Capa Plata) para trazabilidad
        ruta_salida = RUTA_DATOS_PROCESSED / "bank_processed.csv"
        df.to_csv(ruta_salida, index=False)
        logger.info(f"Datos limpios exportados correctamente a: {ruta_salida}")
        
        return df
        
    except Exception as e:
        logger.error(f"Colapso estructural en la transformación de datos: {str(e)}")
        raise e