# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - MODELOS DE CIENCIA DE DATOS
# MAGIC 
# MAGIC **Objetivo:** Construir y entrenar modelos de Machine Learning para segmentación, forecasting y detección de anomalías.
# MAGIC 
# MAGIC **Fuente:** Tablas Gold y Silver
# MAGIC 
# MAGIC **Modelos:**
# MAGIC 1. K-Means Clustering (Segmentación RFM)
# MAGIC 2. Prophet Forecast (Predicción de Ventas)
# MAGIC 3. Isolation Forest (Detección de Anomalías)
# MAGIC 
# MAGIC **Tracking:** MLflow para versionamiento y experimentación

# COMMAND ----------

# Importaciones
import mlflow
import mlflow.sklearn
import mlflow.prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, mean_absolute_percentage_error, mean_squared_error

from prophet import Prophet

print("✓ Librerías importadas")

# Configurar experimento MLflow
mlflow.set_experiment("/ventas_duratex/experimentos/duratex_ventas_2024_2026")
print("✓ Experimento MLflow configurado: /ventas_duratex/experimentos/duratex_ventas_2024_2026")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODELO 1: K-Means Clustering (Segmentación RFM)
# MAGIC 
# MAGIC **Objetivo:** Segmentar clientes usando algoritmo K-Means sobre métricas RFM normalizadas.
# MAGIC 
# MAGIC **Output:** 4 clusters de clientes con características distintivas.

# COMMAND ----------

print("🧠 MODELO 1: K-Means Clustering - Segmentación RFM")
print("=" * 80)

# Leer datos RFM
df_rfm = spark.read.table("ventas_duratex.gold.clientes_rfm").toPandas()

print(f"Total clientes: {len(df_rfm):,}")
print(f"Columnas RFM: RECENCIA, FRECUENCIA, MONTO")

# Preparar datos para clustering
X = df_rfm[['RECENCIA', 'FRECUENCIA', 'MONTO']].values

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✓ Datos escalados con StandardScaler")

# COMMAND ----------

# Elbow Method para determinar K óptimo
print("\nCalculando Elbow Curve...")

inertias = []
K_range = range(2, 9)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)

# Graficar Elbow Curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='#3498db')
ax.set_xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inercia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
ax.set_title('Elbow Method para K-Means\nDeterminación de Número Óptimo de Clusters', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='K=4 (Seleccionado)')
ax.legend()
plt.tight_layout()
plt.show()

print("✓ Elbow curve generada")
print("\n➡️ K óptimo seleccionado: 4 clusters")

# COMMAND ----------

# Entrenar K-Means con K=4
print("\nEntrenando modelo K-Means con K=4...")

K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_scaled)

# Asignar clusters
df_rfm['CLUSTER'] = kmeans.labels_

# Calcular silhouette score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f"\n✓ Modelo entrenado exitosamente")
print(f"✓ Silhouette Score: {sil_score:.4f}")

# COMMAND ----------

# Registrar modelo en MLflow
print("\nRegistrando modelo en MLflow...")

with mlflow.start_run(run_name="kmeans_rfm_segmentacion") as run:
    # Log params
    mlflow.log_param("n_clusters", K_OPTIMAL)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("features", "RECENCIA,FRECUENCIA,MONTO")
    
    # Log metrics
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.log_metric("inertia", kmeans.inertia_)
    
    # Log model
    mlflow.sklearn.log_model(kmeans, "kmeans_model")
    mlflow.sklearn.log_model(scaler, "scaler")
    
    print(f"✓ Run ID: {run.info.run_id}")
    print("✓ Modelo y scaler registrados en MLflow")

# COMMAND ----------

# Visualizar clusters
print("\nVisualizando clusters...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter 1: Frecuencia vs Monto
for cluster in range(K_OPTIMAL):
    mask = df_rfm['CLUSTER'] == cluster
    axes[0].scatter(
        df_rfm.loc[mask, 'FRECUENCIA'],
        df_rfm.loc[mask, 'MONTO']/1e6,  # Millones
        label=f'Cluster {cluster}',
        alpha=0.6,
        s=50
    )

axes[0].set_xlabel('Frecuencia (Número de Compras)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Monto Total (Millones COP)', fontsize=11, fontweight='bold')
axes[0].set_title('Clusters: Frecuencia vs Monto', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Scatter 2: Recencia vs Frecuencia
for cluster in range(K_OPTIMAL):
    mask = df_rfm['CLUSTER'] == cluster
    axes[1].scatter(
        df_rfm.loc[mask, 'RECENCIA'],
        df_rfm.loc[mask, 'FRECUENCIA'],
        label=f'Cluster {cluster}',
        alpha=0.6,
        s=50
    )

axes[1].set_xlabel('Recencia (Días desde última compra)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frecuencia (Número de Compras)', fontsize=11, fontweight='bold')
axes[1].set_title('Clusters: Recencia vs Frecuencia', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualizaciones de clusters generadas")

# COMMAND ----------

# Analizar características de clusters
print("\nCARACTERÍSTICAS DE LOS CLUSTERS:")
print("=" * 80)

cluster_summary = df_rfm.groupby('CLUSTER').agg({
    'CLIENTE': 'count',
    'RECENCIA': 'mean',
    'FRECUENCIA': 'mean',
    'MONTO': 'mean'
}).round(2)

cluster_summary.columns = ['Num_Clientes', 'Recencia_Prom', 'Frecuencia_Prom', 'Monto_Prom']
cluster_summary['Monto_Prom_MM'] = (cluster_summary['Monto_Prom'] / 1e6).round(2)

# Etiquetar clusters basado en características
def etiquetar_cluster(row):
    if row['Recencia_Prom'] < 180 and row['Frecuencia_Prom'] > 10:
        return "Champions"
    elif row['Frecuencia_Prom'] > 5:
        return "Clientes_Leales"
    elif row['Recencia_Prom'] > 365:
        return "Inactivos"
    else:
        return "Potenciales"

cluster_summary['Etiqueta'] = cluster_summary.apply(etiquetar_cluster, axis=1)

display(cluster_summary[['Num_Clientes', 'Recencia_Prom', 'Frecuencia_Prom', 'Monto_Prom_MM', 'Etiqueta']])

print("\n✓ Análisis de clusters completado")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretación de Clusters
# MAGIC 
# MAGIC * **Champions:** Clientes con alta frecuencia y recencia baja (compraron recientemente y compran seguido)
# MAGIC * **Clientes Leales:** Alta frecuencia de compra, pero recencia variable
# MAGIC * **Potenciales:** Frecuencia media, oportunidad de conversión con campañas dirigidas
# MAGIC * **Inactivos:** Alta recencia (no compran hace mucho tiempo), requieren estrategias de reactivación
# MAGIC 
# MAGIC **Recomendaciones de negocio:**
# MAGIC * Champions: Programas de lealtad premium
# MAGIC * Leales: Cross-selling y up-selling
# MAGIC * Potenciales: Campañas de conversión y beneficios
# MAGIC * Inactivos: Campañas de reactivación con descuentos especiales

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODELO 2: Prophet Forecast (Predicción de Ventas)
# MAGIC 
# MAGIC **Objetivo:** Forecasting de ventas mensuales para el mercado interno usando Prophet.
# MAGIC 
# MAGIC **Output:** Predicción de 6 meses futuros con intervalos de confianza.

# COMMAND ----------

print("🔮 MODELO 2: Prophet - Forecast de Ventas")
print("=" * 80)

# Leer datos de ventas por período
df_periodo = spark.read.table("ventas_duratex.gold.ventas_por_periodo") \
    .filter(col("Mercado") == "Interno") \
    .select("FECHA_PERIODO", "VENTA_NETA_TOTAL") \
    .orderBy("FECHA_PERIODO") \
    .toPandas()

print(f"Períodos históricos: {len(df_periodo)}")
print(f"Fecha inicio: {df_periodo['FECHA_PERIODO'].min()}")
print(f"Fecha fin: {df_periodo['FECHA_PERIODO'].max()}")

# Preparar datos para Prophet (requiere columnas 'ds' y 'y')
df_prophet = df_periodo.rename(columns={
    'FECHA_PERIODO': 'ds',
    'VENTA_NETA_TOTAL': 'y'
})

# Dividir en train y test (2024-2025 train, 2026 test)
split_date = '2026-01-01'
df_train = df_prophet[df_prophet['ds'] < split_date]
df_test = df_prophet[df_prophet['ds'] >= split_date]

print(f"\nDatos de entrenamiento: {len(df_train)} períodos (2024-2025)")
print(f"Datos de prueba: {len(df_test)} períodos (2026)")

# COMMAND ----------

# Entrenar modelo Prophet
print("\nEntrenando modelo Prophet...")

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    seasonality_mode='multiplicative',
    interval_width=0.80,
    changepoint_prior_scale=0.05
)

model.fit(df_train)
print("✓ Modelo entrenado exitosamente")

# Generar predicciones
periods_future = 12  # 6 meses de test + 6 meses futuros
future = model.make_future_dataframe(periods=periods_future, freq='MS')
forecast = model.predict(future)

print(f"✓ Predicciones generadas para {periods_future} períodos")

# COMMAND ----------

# Visualizar forecast
print("\nVisualizando predicciones...")

fig1 = model.plot(forecast, figsize=(14, 6))
plt.title('Forecast de Ventas - Mercado Interno Duratex\nHistorial + Predicción 12 Meses', 
          fontsize=14, fontweight='bold')
plt.xlabel('Fecha', fontsize=12, fontweight='bold')
plt.ylabel('Venta Neta (COP)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("✓ Gráfica de forecast generada")

# COMMAND ----------

# Visualizar componentes
print("\nVisualizando componentes del modelo...")

fig2 = model.plot_components(forecast, figsize=(12, 8))
plt.tight_layout()
plt.show()

print("✓ Componentes (tendencia y estacionalidad) graficados")

# COMMAND ----------

# Evaluar modelo en datos de test
print("\nEvaluando modelo en datos 2026 (test)...")

# Extraer predicciones para período de test
forecast_test = forecast[forecast['ds'].isin(df_test['ds'])]

# Calcular métricas
y_true = df_test['y'].values
y_pred = forecast_test['yhat'].values

mape = mean_absolute_percentage_error(y_true, y_pred) * 100
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = np.mean(np.abs(y_true - y_pred))

print(f"\nMÉTRICAS DE EVALUACIÓN (Datos 2026):")
print("=" * 60)
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"RMSE (Root Mean Squared Error):        ${rmse:,.0f} COP")
print(f"MAE (Mean Absolute Error):              ${mae:,.0f} COP")
print("=" * 60)

# COMMAND ----------

# Registrar modelo en MLflow
print("\nRegistrando modelo Prophet en MLflow...")

with mlflow.start_run(run_name="prophet_forecast_ventas_interno") as run:
    # Log params
    mlflow.log_param("mercado", "Interno")
    mlflow.log_param("yearly_seasonality", True)
    mlflow.log_param("seasonality_mode", "multiplicative")
    mlflow.log_param("interval_width", 0.80)
    mlflow.log_param("train_periods", len(df_train))
    mlflow.log_param("forecast_periods", periods_future)
    
    # Log metrics
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    
    # Log model
    mlflow.prophet.log_model(model, "prophet_model")
    
    print(f"✓ Run ID: {run.info.run_id}")
    print("✓ Modelo Prophet registrado en MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interpretación del Forecast
# MAGIC 
# MAGIC * **Tendencia:** Muestra el crecimiento o decrecimiento general del negocio
# MAGIC * **Estacionalidad anual:** Identifica meses de alta y baja demanda
# MAGIC * **Intervalos de confianza:** Representan la incertidumbre de la predicción (80%)
# MAGIC 
# MAGIC **Recomendaciones de negocio:**
# MAGIC * Planificar inventario basado en meses de alta demanda predichos
# MAGIC * Ajustar capacidad operativa según tendencia
# MAGIC * Preparar campañas comerciales en meses de baja demanda estacional

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODELO 3: Isolation Forest (Detección de Anomalías)
# MAGIC 
# MAGIC **Objetivo:** Detectar transacciones anómalas en ventas usando Isolation Forest.
# MAGIC 
# MAGIC **Output:** Identificación de operaciones atípicas para revisión.

# COMMAND ----------

print("🚨 MODELO 3: Isolation Forest - Detección de Anomalías")
print("=" * 80)

# Leer datos transaccionales de Silver (solo ventas)
df_ventas = spark.read.table("ventas_duratex.silver.ventas_limpias") \
    .filter(col("TIPO_TRANSACCION") == "VENTA") \
    .select(
        "NUMERO_FACTURA",
        "CLIENTE",
        "VLR_VENTA_NETA",
        "CANT_FACTURADA",
        "PRECIO_UNITARIO",
        "VLR_DSCTO_LIN",
        "COSTO_PROMEDIO",
        "PCT_MARGEN"
    ) \
    .toPandas()

print(f"Total transacciones analizadas: {len(df_ventas):,}")

# COMMAND ----------

# Preparar features para detección
features = [
    'VLR_VENTA_NETA',
    'CANT_FACTURADA',
    'PRECIO_UNITARIO',
    'VLR_DSCTO_LIN',
    'COSTO_PROMEDIO'
]

X_anom = df_ventas[features].fillna(0)

# Escalar
scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

print(f"\nFeatures utilizados: {', '.join(features)}")
print("✓ Datos preparados y escalados")

# COMMAND ----------

# Entrenar Isolation Forest
print("\nEntrenando modelo Isolation Forest...")

CONTAMINATION = 0.02  # Esperamos 2% de anomalías

iso_forest = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42,
    n_estimators=100,
    max_samples='auto'
)

# Predecir: -1 = anomalía, 1 = normal
predictions = iso_forest.fit_predict(X_anom_scaled)

df_ventas['ES_ANOMALIA'] = predictions == -1

n_anomalias = df_ventas['ES_ANOMALIA'].sum()
porc_anomalias = (n_anomalias / len(df_ventas)) * 100

print(f"\n✓ Modelo entrenado exitosamente")
print(f"✓ Anomalías detectadas: {n_anomalias:,} ({porc_anomalias:.2f}%)")

# COMMAND ----------

# Registrar modelo en MLflow
print("\nRegistrando modelo en MLflow...")

with mlflow.start_run(run_name="isolation_forest_anomalias") as run:
    # Log params
    mlflow.log_param("contamination", CONTAMINATION)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("features", ",".join(features))
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("n_anomalias", n_anomalias)
    mlflow.log_metric("porcentaje_anomalias", porc_anomalias)
    
    # Log model
    mlflow.sklearn.log_model(iso_forest, "isolation_forest_model")
    mlflow.sklearn.log_model(scaler_anom, "scaler_anomalias")
    
    print(f"✓ Run ID: {run.info.run_id}")
    print("✓ Modelo Isolation Forest registrado en MLflow")

# COMMAND ----------

# Analizar anomalías detectadas
print("\nANALIZANDO ANOMALÍAS DETECTADAS:")
print("=" * 80)

# Top 20 anomalías por valor de venta
df_anomalias = df_ventas[df_ventas['ES_ANOMALIA'] == True].copy()
df_anomalias_top = df_anomalias.nlargest(20, 'VLR_VENTA_NETA')

print(f"\nTop 20 anomalías por valor de venta:")
display(df_anomalias_top[[
    'NUMERO_FACTURA',
    'CLIENTE',
    'VLR_VENTA_NETA',
    'CANT_FACTURADA',
    'PRECIO_UNITARIO',
    'VLR_DSCTO_LIN',
    'PCT_MARGEN'
]].head(20))

# COMMAND ----------

# Visualizar anomalías
print("\nVisualizando anomalías...")

fig, ax = plt.subplots(figsize=(14, 8))

# Scatter: normales vs anomalías
normales = df_ventas[df_ventas['ES_ANOMALIA'] == False]
anomalias = df_ventas[df_ventas['ES_ANOMALIA'] == True]

ax.scatter(
    normales['PRECIO_UNITARIO'],
    normales['VLR_VENTA_NETA']/1e6,
    alpha=0.3,
    s=10,
    color='blue',
    label=f'Normal ({len(normales):,})'
)

ax.scatter(
    anomalias['PRECIO_UNITARIO'],
    anomalias['VLR_VENTA_NETA']/1e6,
    alpha=0.8,
    s=50,
    color='red',
    marker='x',
    linewidths=2,
    label=f'Anomalía ({len(anomalias):,})'
)

ax.set_xlabel('Precio Unitario (COP)', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor Venta Neta (Millones COP)', fontsize=12, fontweight='bold')
ax.set_title('Detección de Anomalías en Transacciones de Venta\nIsolation Forest',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualización de anomalías generada")

# COMMAND ----------

# Estadísticas de anomalías
print("\nESTADÍSTICAS COMPARATIVAS:")
print("=" * 80)

stats_comp = pd.DataFrame({
    'Tipo': ['Normal', 'Anomalía'],
    'Count': [len(normales), len(anomalias)],
    'VLR_VENTA_Media': [
        normales['VLR_VENTA_NETA'].mean(),
        anomalias['VLR_VENTA_NETA'].mean()
    ],
    'PRECIO_UNIT_Media': [
        normales['PRECIO_UNITARIO'].mean(),
        anomalias['PRECIO_UNITARIO'].mean()
    ],
    'DESCUENTO_Media': [
        normales['VLR_DSCTO_LIN'].mean(),
        anomalias['VLR_DSCTO_LIN'].mean()
    ],
    'MARGEN_Media': [
        normales['PCT_MARGEN'].mean(),
        anomalias['PCT_MARGEN'].mean()
    ]
}).round(2)

display(stats_comp)

print("\n✓ Análisis de anomalías completado")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgos y Recomendaciones - Anomalías
# MAGIC 
# MAGIC **Posibles causas de anomalías:**
# MAGIC * 🔴 Precios excesivamente altos o bajos
# MAGIC * 🔴 Descuentos atípicos (muy altos o fuera de política)
# MAGIC * 🔴 Cantidades inusuales (pedidos muy grandes o muy pequeños)
# MAGIC * 🔴 Márgenes negativos o extraordinariamente altos
# MAGIC * 🔴 Errores de captura de datos
# MAGIC 
# MAGIC **Recomendaciones:**
# MAGIC 1. Revisar manualmente las top 50 anomalías
# MAGIC 2. Validar políticas de descuentos con equipo comercial
# MAGIC 3. Implementar alertas automáticas para transacciones anómalas
# MAGIC 4. Actualizar modelo mensualmente con nuevos datos

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🎆 Resumen Final - Modelos de Ciencia de Datos
# MAGIC 
# MAGIC ### Modelos Entrenados y Registrados en MLflow
# MAGIC 
# MAGIC #### 1. K-Means Clustering (Segmentación RFM)
# MAGIC * ✅ **Clusters:** 4 segmentos de clientes
# MAGIC * ✅ **Métrica:** Silhouette Score
# MAGIC * ✅ **Output:** Champions, Leales, Potenciales, Inactivos
# MAGIC * 🎯 **Uso:** Estrategias de CRM y marketing personalizado
# MAGIC 
# MAGIC #### 2. Prophet Forecast (Predicción de Ventas)
# MAGIC * ✅ **Horizonte:** 12 meses futuros
# MAGIC * ✅ **Métricas:** MAPE, RMSE, MAE
# MAGIC * ✅ **Componentes:** Tendencia + Estacionalidad anual
# MAGIC * 🎯 **Uso:** Planificación de inventario y capacidad
# MAGIC 
# MAGIC #### 3. Isolation Forest (Detección de Anomalías)
# MAGIC * ✅ **Anomalías detectadas:** ~2% de transacciones
# MAGIC * ✅ **Features:** Precio, cantidad, descuento, costo, margen
# MAGIC * ✅ **Output:** Transacciones atípicas para revisión
# MAGIC * 🎯 **Uso:** Control de calidad y detección de fraudes
# MAGIC 
# MAGIC ### Experimentos MLflow
# MAGIC * 📁 **Ubicación:** `/ventas_duratex/experimentos/duratex_ventas_2024_2026`
# MAGIC * 📦 **Modelos registrados:** 3 (K-Means, Prophet, Isolation Forest)
# MAGIC * 📊 **Métricas trackeadas:** Silhouette, MAPE, RMSE, % anomalías
# MAGIC 
# MAGIC ### Próximos Pasos
# MAGIC 1. 🔄 Reentrenar modelos mensualmente con datos actualizados
# MAGIC 2. 🚨 Implementar sistema de alertas para anomalías en tiempo real
# MAGIC 3. 📊 Construir dashboard interactivo con predicciones y segmentos
# MAGIC 4. 🎯 Integrar modelos en pipelines de producción
# MAGIC 
# MAGIC ---
# MAGIC **Proyecto:** Lakehouse Ventas Duratex Colombia  
# MAGIC **Módulo:** Ciencia de Datos y Machine Learning  
# MAGIC **Última actualización:** 2026