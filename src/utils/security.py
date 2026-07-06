import pandas as pd
import hashlib
from src.utils.logger import logger

def anonimizar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica hashing criptográfico y enmascaramiento para cumplir con
    la normativa de protección de datos personales antes de subir a la nube.
    """
    logger.info("🔒 Iniciando protocolo de seguridad: Anonimización de datos...")
    df_seguro = df.copy()
    
    try:
        # 1. Creación de Identificador Anónimo (Hashing SHA-256)
        # Simulamos que cada fila es un cliente y le asignamos un hash irreversible
        # basado en su índice temporal, asegurando que no haya trazabilidad directa.
        df_seguro['client_token'] = df_seguro.index.astype(str).map(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
        
        # 2. Enmascaramiento de Variable Sensible (Edad)
        # Transformamos el dato exacto en un rango para evitar re-identificación
        bins_edad = [17, 25, 35, 45, 55, 65, 100]
        labels_edad = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        df_seguro['age_group'] = pd.cut(df_seguro['age'], bins=bins_edad, labels=labels_edad)
        
        # 3. Eliminación del dato personal crudo
        # Destruimos la columna 'age' exacta de la memoria antes de la nube
        df_seguro = df_seguro.drop(columns=['age'])
        
        # Reordenar para que el token quede primero
        cols = ['client_token'] + [col for col in df_seguro.columns if col != 'client_token']
        df_seguro = df_seguro[cols]
        
        logger.info("🔒 Datos anonimizados exitosamente. Listo para tránsito seguro a Supabase.")
        return df_seguro

    except Exception as e:
        logger.error(f"Fallo crítico en el encriptado de datos: {str(e)}")
        raise e