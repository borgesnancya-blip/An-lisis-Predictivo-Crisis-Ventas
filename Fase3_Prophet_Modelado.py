import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt 
import logging

# Silenciar mensajes de advertencia de Prophet
logging.getLogger('prophet').setLevel(logging.WARNING)

# 1. Carga de datos (Ajusta la ruta)
df = pd.read_csv('C:/Users/Outlet/Downloads/data.csv')

# --- CONFIGURACIÓN DE COLUMNAS ---
NOMBRE_COLUMNA_INGRESO = 'Ingreso Total'
NOMBRE_COLUMNA_AÑO = 'Año'
NOMBRE_COLUMNA_MES = 'Mes'
# --------------------------------

# 2. Limpieza de Ingreso ('y'): Elimina símbolos, reemplaza coma decimal por punto y convierte a numérico.
df['y'] = (
    df[NOMBRE_COLUMNA_INGRESO].astype(str)
    .str.replace('$', '', regex=False)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
)
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# 3. Construcción del Índice de Tiempo ('ds')
# Diccionario para mapear nombres de meses a números.
mapeo_meses = {
   'ene' : '01', 'feb' : '02', 'mar' : '03', 'abr' : '04', 'may' : '05', 'jun' : '06',
   'jul' : '07', 'ago' : '08', 'sep' : '09', 'oct' : '10', 'nov' : '11', 'dic' : '12'    
}
df['Mes_Num'] = df[NOMBRE_COLUMNA_MES].astype(str).str.lower().str.strip().map(mapeo_meses)

# Se construye la fecha en formato 'YYYY-MM-01' y se convierte a datetime.
df['ds'] = df[NOMBRE_COLUMNA_AÑO].astype(str) + '-' + df['Mes_Num'] + '-01'
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# 4. Limpieza y Preparación Final del DataFrame (ds, y)
df = df[['ds', 'y']].dropna()
df = df.drop_duplicates(subset= ['ds'], keep= 'first')
df = df.sort_values(by= 'ds').reset_index(drop=True)

# 5. Inicialización y Entrenamiento del Modelo Prophet
modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
modelo.fit(df)
print("✅ Modelo Prophet entrenado con éxito.")

# 6. Predicción a Futuro (6 Meses)
futuro = modelo.make_future_dataframe(periods=6, freq='MS')
forecast = modelo.predict(futuro)

# 7. Extracción de la Predicción Ejecutiva (Últimas 6 filas)
prediccion_futura = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)

print("\n--- Predicción de Ingresos (Próximos 6 meses) ---")
print(prediccion_futura.round(2))

# 8. Visualización y Guardado del Gráfico (Evidencia Ejecutiva)
fig = modelo.plot(forecast)
plt.title('Proyección de Ingresos: Identificación de la Crisis (Prophet)')
plt.ylabel('Ingreso Total Proyectado ($)')
plt.xlabel('Fecha')
plt.tight_layout()

# Se guarda el gráfico en alta resolución para el repositorio.
fig.savefig('prediccion_prophet_FINAL_CORREGIDA.png', dpi=300)
print("\n✅ Gráfico de predicción guardado como 'prediccion_prophet_FINAL_CORREGIDA.png'.")

plt.show() # Mostrar el gráfico al ejecutar





