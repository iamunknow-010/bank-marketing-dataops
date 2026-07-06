import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Dashboard Bank Marketing - DataOps", layout="wide")

st.title("📊 Dashboard de Predicción y Rendimiento - Campaña Bancaria")
st.markdown("Este panel interactivo permite evaluar el comportamiento del modelo de IA y simular predicciones en tiempo real para la contratación de depósitos a plazo.")

# 2. CARGA DE DATOS Y MODELO
@st.cache_resource
def cargar_recursos():
    df = pd.read_csv('data/processed/bank_processed.csv')
    with open('models/modelo_bank_marketing.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return df, modelo

try:
    df, modelo = cargar_recursos()
    st.success("✅ Modelo Random Forest e historial de datos cargados exitosamente desde el pipeline.")
except Exception as e:
    st.error(f"❌ Error al cargar los archivos: {e}.")
    st.stop()

# 3. DISEÑO DE PESTAÑAS
tab1, tab2 = st.tabs(["📈 Análisis de Rendimiento y Datos", "🔮 Simulador de Predicciones (IA)"])

# ----------------------------------------------------
# PESTAÑA 1: ANÁLISIS DE RENDIMIENTO Y DATOS
# ----------------------------------------------------
with tab1:
    st.header("Análisis del Perfil de Clientes e Historial")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clientes Analizados", f"{len(df):,}")
    with col2:
        Tasa_conversion = (df['deposit'].sum() / len(df)) * 100
        st.metric("Tasa de Conversión Real", f"{Tasa_conversion:.1f}%")
    with col3:
        st.metric("Modelo Ganador Desplegado", "Random Forest (.pkl)")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Aceptación de depósito según Educación")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='education', hue='deposit', palette='Set2', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col_g2:
        st.subheader("Distribución del Balance según Decisión")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='deposit', y='balance', palette='pastel', ax=ax)
        ax.set_ylim(-1000, 5000)
        st.pyplot(fig)

# ----------------------------------------------------
# PESTAÑA 2: SIMULADOR DE PREDICCIONES (MACHINE LEARNING REAL)
# ----------------------------------------------------
with tab2:
    st.header("Simulador de Clientes en Tiempo Real")
    st.markdown("Ingresa los datos del cliente para evaluar la probabilidad estadística de contratación mediante Inteligencia Artificial.")
    
    # Formulario completo con todas las variables usadas en el entrenamiento
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Datos Personales**")
        age = st.slider("Edad", 18, 95, 35)
        job = st.selectbox("Profesión", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'self-employed', 'student'])
        marital = st.selectbox("Estado Civil", ['married', 'single', 'divorced'])
        education = st.selectbox("Educación", ['secondary', 'tertiary', 'primary'])
        
    with col2:
        st.markdown("**💰 Perfil Financiero**")
        balance = st.number_input("Balance de cuenta (Euros)", value=1500)
        default = st.selectbox("¿Crédito en mora?", [0, 1], format_func=lambda x: "Sí" if x==1 else "No")
        housing = st.selectbox("¿Préstamo hipotecario?", [0, 1], format_func=lambda x: "Sí" if x==1 else "No")
        loan = st.selectbox("¿Préstamo personal?", [0, 1], format_func=lambda x: "Sí" if x==1 else "No")
        
    with col3:
        st.markdown("**📞 Historial de Contacto**")
        contact = st.selectbox("Canal de contacto", ['cellular', 'telephone'])
        month = st.selectbox("Mes", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day = st.slider("Día del mes", 1, 31, 15)
        duration = st.slider("Duración última llamada (seg)", 0, 3000, 250)
        campaign = st.slider("Contactos esta campaña", 1, 10, 1)
        pdays = st.number_input("Días desde última campaña (-1=nunca)", value=-1)
        previous = st.slider("Contactos previos", 0, 10, 0)
        poutcome = st.selectbox("Resultado campaña anterior", ['success', 'failure', 'other'])

    st.markdown("---")
    st.info("🔒 Los datos ingresados son inyectados al modelo matemático Random Forest para calcular la probabilidad exacta.")
    
    if st.button("🚀 Ejecutar Predicción con Inteligencia Artificial"):
        try:
            # 1. Crear dataframe con la entrada del usuario (1 fila)
            user_data = pd.DataFrame({
                'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
                'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan],
                'contact': [contact], 'day': [day], 'month': [month], 'duration': [duration],
                'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome]
            })
            
            # 2. Truco de Ingeniería: Concatenar con los datos base para asegurar las 42 columnas tras el get_dummies
            X_base = df.drop(columns=['deposit']) # Las características originales
            X_combined = pd.concat([X_base, user_data], ignore_index=True)
            
            # 3. Aplicar One-Hot Encoding tal como en el notebook
            categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
            X_encoded = pd.get_dummies(X_combined, columns=categorical_cols, drop_first=True, dtype=int)
            
            # 4. Extraer solo la última fila (los datos transformados del usuario)
            X_user_encoded = X_encoded.iloc[[-1]]
            
            # 5. PREDICCIÓN MATEMÁTICA CON EL MODELO .PKL
            prediccion = modelo.predict(X_user_encoded)[0]
            probabilidad = modelo.predict_proba(X_user_encoded)[0][1] * 100 
            
            # 6. Mostrar el resultado estadístico
            st.subheader("Resultado de la Auditoría del Modelo")
            if prediccion == 1:
                st.success(f"### ¡ALTA PROBABILIDAD DE CONTRATACIÓN! ({probabilidad:.1f}%)")
                st.balloons()
                st.markdown("👉 **Acción sugerida:** Priorizar a este cliente en la lista de llamadas de la jornada.")
            else:
                st.error(f"### BAJA PROBABILIDAD DE CONTRATACIÓN ({probabilidad:.1f}%)")
                st.markdown("👉 **Acción sugerida:** Evitar contacto telefónico directo para reducir costos operativos y fatiga comercial.")
                
        except Exception as e:
            st.error(f"Falla crítica en la predicción. Detalle técnico: {e}")