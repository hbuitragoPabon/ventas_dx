# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - ANÁLISIS Y VISUALIZACIONES
# MAGIC 
# MAGIC **Objetivo:** Generar visualizaciones avanzadas para análisis exploratorio y soporte a decisiones de negocio.
# MAGIC 
# MAGIC **Fuente:** Tablas Gold (`ventas_duratex.gold.*`)
# MAGIC 
# MAGIC **Visualizaciones:**
# MAGIC 1. Tendencia de Ventas Mensual (line plot doble eje)
# MAGIC 2. Participación por Línea de Producto (donut chart)
# MAGIC 3. Top 10 Vendedores (barras horizontales coloreadas por margen)
# MAGIC 4. Estacionalidad de Ventas (heatmap)
# MAGIC 5. Ventas vs Devoluciones por Vendedor (barras agrupadas + línea)
# MAGIC 6. Distribución de Márgenes por Sublínea (boxplot)
# MAGIC 7. Segmentación RFM (scatter interactivo)
# MAGIC 8. Comparativo YoY por Bodega (barras agrupadas)

# COMMAND ----------

# Importaciones generales
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

print("✓ Librerías importadas")
print("✓ Estilos configurados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 1: Tendencia de Ventas Mensual
# MAGIC 
# MAGIC **Análisis:** Evolución temporal de ventas netas y margen promedio por mercado (Interno/Externo) con media móvil de 3 meses.
# MAGIC 
# MAGIC **Conclusión de negocio:** Permite identificar tendencias, estacionalidad y oportunidades de crecimiento por mercado.

# COMMAND ----------

print("📈 GRÁFICA 1: Tendencia de Ventas Mensual")
print("=" * 80)

# Leer datos
df_periodo = spark.read.table("ventas_duratex.gold.ventas_por_periodo").toPandas()

# Convertir fecha a datetime
df_periodo['FECHA_PERIODO'] = pd.to_datetime(df_periodo['FECHA_PERIODO'])

# Ordenar por fecha y mercado
df_periodo = df_periodo.sort_values(['FECHA_PERIODO', 'Mercado'])

# Calcular media móvil de 3 meses por mercado
df_periodo['VENTA_MA3'] = df_periodo.groupby('Mercado')['VENTA_NETA_TOTAL'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Crear figura con doble eje Y
fig, ax1 = plt.subplots(figsize=(14, 6))

# Eje Y izquierdo: Ventas
colores_mercado = {'Interno': '#1f77b4', 'Externo': '#ff7f0e'}

for mercado in df_periodo['Mercado'].unique():
    data = df_periodo[df_periodo['Mercado'] == mercado]
    # Línea de ventas
    ax1.plot(data['FECHA_PERIODO'], data['VENTA_NETA_TOTAL']/1e9, 
             label=f'{mercado} - Ventas', color=colores_mercado.get(mercado, 'gray'),
             linewidth=2, marker='o', markersize=4)
    # Media móvil
    ax1.plot(data['FECHA_PERIODO'], data['VENTA_MA3']/1e9, 
             linestyle='--', alpha=0.6, color=colores_mercado.get(mercado, 'gray'),
             linewidth=1.5, label=f'{mercado} - MA(3)')

ax1.set_xlabel('Período', fontsize=12, fontweight='bold')
ax1.set_ylabel('Venta Neta (Miles de Millones COP)', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)

# Eje Y derecho: Margen
ax2 = ax1.twinx()
for mercado in df_periodo['Mercado'].unique():
    data = df_periodo[df_periodo['Mercado'] == mercado]
    ax2.plot(data['FECHA_PERIODO'], data['MARGEN_PROMEDIO_PCT'], 
             linestyle=':', alpha=0.5, color=colores_mercado.get(mercado, 'gray'),
             linewidth=2, label=f'{mercado} - Margen %')

ax2.set_ylabel('Margen Promedio (%)', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y')

# Título y leyendas
plt.title('Tendencia de Ventas Netas Duratex Colombia 2024-2026\n(Con Media Móvil 3 Meses)', 
          fontsize=14, fontweight='bold', pad=20)

# Combinar leyendas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.show()

print("✓ Gráfica generada exitosamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 2: Participación por Línea de Producto
# MAGIC 
# MAGIC **Análisis:** Distribución porcentual de ventas por línea de producto (MDP vs MDF).
# MAGIC 
# MAGIC **Conclusión de negocio:** Identifica las líneas de mayor contribución al revenue total.

# COMMAND ----------

print("🍩 GRÁFICA 2: Participación por Línea de Producto")
print("=" * 80)

# Leer y agrupar por línea
df_producto = spark.read.table("ventas_duratex.gold.ventas_por_producto")
df_linea = df_producto.groupBy("LINEA") \
    .agg({'VENTA_NETA_TOTAL': 'sum'}) \
    .toPandas()

df_linea.columns = ['LINEA', 'VENTA_TOTAL']
df_linea = df_linea[df_linea['LINEA'].notna()]
df_linea = df_linea.sort_values('VENTA_TOTAL', ascending=False)

# Calcular porcentajes
total_ventas = df_linea['VENTA_TOTAL'].sum()
df_linea['PORCENTAJE'] = (df_linea['VENTA_TOTAL'] / total_ventas) * 100

# Colores personalizados
colores = {'MDP': '#CD7F32', 'MDF': '#A8A9AD'}  # Bronce y Plata
colors = [colores.get(linea, '#cccccc') for linea in df_linea['LINEA']]

# Crear donut chart
fig, ax = plt.subplots(figsize=(10, 8))

wedges, texts, autotexts = ax.pie(
    df_linea['VENTA_TOTAL'],
    labels=df_linea['LINEA'],
    autopct=lambda pct: f'{pct:.1f}%\n(${pct*total_ventas/100/1e9:.2f}B)',
    startangle=90,
    colors=colors,
    wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)

# Estilo de textos
for autotext in autotexts:
    autotext.set_color('white')

# Centro del donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Texto central
ax.text(0, 0, f'${total_ventas/1e9:.1f}B\nCOP', 
        horizontalalignment='center', verticalalignment='center',
        fontsize=18, fontweight='bold', color='#333333')

plt.title('Participación de Ventas por Línea de Producto\nDuratex Colombia 2024-2026', 
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("✓ Donut chart generado exitosamente")
print(f"\nTotal ventas: ${total_ventas/1e9:.2f} mil millones COP")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 3: Top 10 Vendedores
# MAGIC 
# MAGIC **Análisis:** Ranking de vendedores por venta neta total, coloreado por margen promedio.
# MAGIC 
# MAGIC **Conclusión de negocio:** Identifica vendedores estrella y aquellos que requieren capacitación para mejorar márgenes.

# COMMAND ----------

print("🏆 GRÁFICA 3: Top 10 Vendedores por Venta y Margen")
print("=" * 80)

# Leer y agrupar por vendedor
df_vendedor = spark.read.table("ventas_duratex.gold.ventas_por_vendedor")
df_vend_agg = df_vendedor.groupBy("NOMBRE_VENDEDOR") \
    .agg(
        {'VENTA_NETA_TOTAL': 'sum', 
         'MARGEN_PROMEDIO_PCT': 'avg'}
    ) \
    .toPandas()

df_vend_agg.columns = ['NOMBRE_VENDEDOR', 'VENTA_TOTAL', 'MARGEN_PROMEDIO']
df_vend_agg = df_vend_agg.sort_values('VENTA_TOTAL', ascending=True).tail(10)

# Crear gráfica de barras horizontales
fig, ax = plt.subplots(figsize=(12, 8))

# Barras coloreadas por margen
bars = ax.barh(
    df_vend_agg['NOMBRE_VENDEDOR'],
    df_vend_agg['VENTA_TOTAL']/1e9,
    color=plt.cm.YlOrRd(df_vend_agg['MARGEN_PROMEDIO']/df_vend_agg['MARGEN_PROMEDIO'].max()),
    edgecolor='black',
    linewidth=0.5
)

# Añadir valores al final de las barras
for i, (venta, margen) in enumerate(zip(df_vend_agg['VENTA_TOTAL'], df_vend_agg['MARGEN_PROMEDIO'])):
    ax.text(venta/1e9 + 0.5, i, f'${venta/1e9:.1f}B\n({margen:.1f}%)', 
            va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Venta Neta Total (Miles de Millones COP)', fontsize=12, fontweight='bold')
ax.set_ylabel('Vendedor', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Vendedores por Venta Neta y Margen Promedio\nDuratex Colombia 2024-2026',
             fontsize=14, fontweight='bold', pad=20)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                            norm=plt.Normalize(vmin=df_vend_agg['MARGEN_PROMEDIO'].min(), 
                                              vmax=df_vend_agg['MARGEN_PROMEDIO'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Margen Promedio (%)', fontsize=11, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ Gráfica de vendedores generada exitosamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 4: Estacionalidad de Ventas (Heatmap)
# MAGIC 
# MAGIC **Análisis:** Mapa de calor mostrando ventas por mes y año para identificar patrones estacionales.
# MAGIC 
# MAGIC **Conclusión de negocio:** Detecta temporadas altas y bajas para planificar inventario y campañas.

# COMMAND ----------

print("🔥 GRÁFICA 4: Estacionalidad de Ventas - Mapa de Calor")
print("=" * 80)

# Leer datos por período
df_periodo = spark.read.table("ventas_duratex.gold.ventas_por_periodo")
df_estac = df_periodo.groupBy("ANO_PERIODO", "MES_PERIODO") \
    .agg({'VENTA_NETA_TOTAL': 'sum'}) \
    .toPandas()

df_estac.columns = ['ANO', 'MES', 'VENTA']

# Pivotar: filas=Año, columnas=Mes
df_pivot = df_estac.pivot(index='ANO', columns='MES', values='VENTA')
df_pivot = df_pivot.fillna(0) / 1e9  # Convertir a miles de millones

# Nombres de meses en español
meses_esp = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
df_pivot.columns = [meses_esp[int(m)-1] for m in df_pivot.columns]

# Crear heatmap
fig, ax = plt.subplots(figsize=(14, 6))

sns.heatmap(
    df_pivot,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Ventas (Miles de Millones COP)'},
    ax=ax,
    vmin=0
)

ax.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax.set_ylabel('Año', fontsize=12, fontweight='bold')
ax.set_title('Estacionalidad de Ventas — Mapa de Calor Mes × Año\nDuratex Colombia 2024-2026',
             fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("✓ Heatmap de estacionalidad generado exitosamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 5: Ventas vs Devoluciones por Vendedor
# MAGIC 
# MAGIC **Análisis:** Comparativo de ventas y devoluciones con tasa de devolución en eje secundario.
# MAGIC 
# MAGIC **Conclusión de negocio:** Identifica vendedores con altas tasas de devolución que requieren atención.

# COMMAND ----------

print("⚖️ GRÁFICA 5: Ventas vs Devoluciones por Vendedor")
print("=" * 80)

# Leer y agrupar
df_vendedor = spark.read.table("ventas_duratex.gold.ventas_por_vendedor")
df_vend_dev = df_vendedor.groupBy("NOMBRE_VENDEDOR") \
    .agg(
        {'VENTA_NETA_TOTAL': 'sum',
         'MONTO_DEVOLUCIONES': 'sum',
         'TASA_DEVOLUCION': 'avg'}
    ) \
    .toPandas()

df_vend_dev.columns = ['NOMBRE_VENDEDOR', 'VENTAS', 'DEVOLUCIONES', 'TASA_DEV']
df_vend_dev = df_vend_dev.sort_values('VENTAS', ascending=False).head(10)
df_vend_dev['DEVOLUCIONES'] = df_vend_dev['DEVOLUCIONES'].abs()  # Valor absoluto

# Crear figura
fig, ax1 = plt.subplots(figsize=(14, 8))

x = np.arange(len(df_vend_dev))
width = 0.35

# Barras de ventas y devoluciones
ax1.bar(x - width/2, df_vend_dev['VENTAS']/1e9, width, 
        label='Ventas', color='#2ecc71', edgecolor='black', linewidth=0.5)
ax1.bar(x + width/2, df_vend_dev['DEVOLUCIONES']/1e9, width, 
        label='Devoluciones', color='#e74c3c', edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Vendedor', fontsize=12, fontweight='bold')
ax1.set_ylabel('Monto (Miles de Millones COP)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df_vend_dev['NOMBRE_VENDEDOR'], rotation=45, ha='right')
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Eje secundario: Tasa de devolución
ax2 = ax1.twinx()
ax2.plot(x, df_vend_dev['TASA_DEV'], 
         color='red', marker='o', linestyle='--', linewidth=2, 
         markersize=8, label='Tasa Devolución (%)')

# Línea de referencia (promedio)
promedio_tasa = df_vend_dev['TASA_DEV'].mean()
ax2.axhline(y=promedio_tasa, color='darkred', linestyle=':', linewidth=1.5, 
            alpha=0.7, label=f'Promedio ({promedio_tasa:.2f}%)')

ax2.set_ylabel('Tasa de Devolución (%)', fontsize=12, fontweight='bold', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

plt.title('Comparativo Ventas vs Devoluciones por Vendedor\nDuratex Colombia 2024-2026',
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

print("✓ Gráfica comparativa generada exitosamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 6: Distribución de Márgenes por Sublínea
# MAGIC 
# MAGIC **Análisis:** Boxplot mostrando la distribución de márgenes por sublínea de producto.
# MAGIC 
# MAGIC **Conclusión de negocio:** Identifica sublíneas con márgenes consistentes vs volátiles y productos problemáticos.

# COMMAND ----------

print("📦 GRÁFICA 6: Distribución de Márgenes por Sublínea")
print("=" * 80)

# Leer datos de Silver (nivel transaccional para boxplot)
df_silver = spark.read.table("ventas_duratex.silver.ventas_limpias") \
    .filter((col("TIPO_TRANSACCION") == "VENTA") & (col("COSTO_INVALIDO") == False)) \
    .select("LINEA", "SUBLINEA", "PCT_MARGEN") \
    .toPandas()

df_silver = df_silver[df_silver['SUBLINEA'].notna()]

# Filtrar outliers extremos (fuera de 3 desviaciones estándar)
q1 = df_silver['PCT_MARGEN'].quantile(0.01)
q99 = df_silver['PCT_MARGEN'].quantile(0.99)
df_silver = df_silver[(df_silver['PCT_MARGEN'] >= q1) & (df_silver['PCT_MARGEN'] <= q99)]

# Crear boxplot
fig, ax = plt.subplots(figsize=(14, 8))

sns.boxplot(
    data=df_silver,
    x='SUBLINEA',
    y='PCT_MARGEN',
    hue='LINEA',
    palette={'MDP': '#CD7F32', 'MDF': '#A8A9AD'},
    ax=ax
)

# Línea de referencia en 0
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Margen = 0%')

ax.set_xlabel('Sublínea de Producto', fontsize=12, fontweight='bold')
ax.set_ylabel('Margen (%)', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Margen (%) por Sublínea de Producto\nDuratex Colombia 2024-2026',
             fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
ax.legend(title='Línea', loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Boxplot de márgenes generado exitosamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 7: Segmentación RFM (Scatter Interactivo)
# MAGIC 
# MAGIC **Análisis:** Visualización interactiva de clientes en dimensiones RFM con segmentación automática.
# MAGIC 
# MAGIC **Conclusión de negocio:** Base para estrategias de CRM, retención y marketing personalizado.

# COMMAND ----------

print("🎯 GRÁFICA 7: Segmentación RFM de Clientes")
print("=" * 80)

# Leer datos RFM
df_rfm = spark.read.table("ventas_duratex.gold.clientes_rfm").toPandas()

# Scatter interactivo con Plotly
fig = px.scatter(
    df_rfm,
    x='FRECUENCIA',
    y='MONTO',
    size='RECENCIA',
    size_max=20,
    color='SEGMENTO_RFM',
    hover_name='CLIENTE',
    hover_data={
        'R_score': True,
        'F_score': True,
        'M_score': True,
        'RFM_SCORE': True,
        'MONTO': ':,.0f',
        'FRECUENCIA': True,
        'RECENCIA': True
    },
    color_discrete_map={
        'Champion': '#2ecc71',
        'Leal': '#3498db',
        'Potencial': '#f39c12',
        'En_Riesgo': '#e74c3c',
        'Inactivo': '#95a5a6',
        'Regular': '#9b59b6'
    },
    title='Segmentación RFM de Clientes Duratex<br><sub>Tamaño = Recencia (menor es mejor) | Color = Segmento</sub>',
    labels={
        'FRECUENCIA': 'Frecuencia (Número de Compras)',
        'MONTO': 'Monto Total (COP)',
        'RECENCIA': 'Recencia (Días desde última compra)'
    }
)

fig.update_layout(
    height=600,
    font=dict(size=12),
    hovermode='closest'
)

fig.show()

print("✓ Scatter RFM interactivo generado exitosamente")

# Tabla resumen de segmentos
print("\nDistribución de clientes por segmento:")
segmento_summary = df_rfm.groupby('SEGMENTO_RFM').agg({
    'CLIENTE': 'count',
    'MONTO': 'sum',
    'FRECUENCIA': 'mean',
    'RECENCIA': 'mean'
}).round(2)
segmento_summary.columns = ['Num_Clientes', 'Venta_Total', 'Frecuencia_Prom', 'Recencia_Prom']
segmento_summary = segmento_summary.sort_values('Venta_Total', ascending=False)
display(segmento_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GRÁFICA 8: Comparativo YoY por Bodega
# MAGIC 
# MAGIC **Análisis:** Comparación de ventas anuales por bodega/CEDI para identificar crecimiento y contracciones.
# MAGIC 
# MAGIC **Conclusión de negocio:** Evaluar desempeño logístico y regional para optimizar distribución.

# COMMAND ----------

print("🏭 GRÁFICA 8: Comparativo Anual de Ventas por Bodega")
print("=" * 80)

# Leer datos de bodegas
df_bodega = spark.read.table("ventas_duratex.gold.ventas_por_bodega")
df_bodega_ano = df_bodega.groupBy("DESCRIPCION_BODEGA", "ANO_PERIODO") \
    .agg({'VENTA_NETA_TOTAL': 'sum'}) \
    .toPandas()

df_bodega_ano.columns = ['BODEGA', 'ANO', 'VENTA']

# Top 10 bodegas por venta 2025
top_bodegas_2025 = df_bodega_ano[df_bodega_ano['ANO'] == 2025] \
    .nlargest(10, 'VENTA')['BODEGA'].tolist()

df_top = df_bodega_ano[df_bodega_ano['BODEGA'].isin(top_bodegas_2025)]

# Pivotar
df_pivot = df_top.pivot(index='BODEGA', columns='ANO', values='VENTA').fillna(0) / 1e9

# Ordenar por venta 2025
if 2025 in df_pivot.columns:
    df_pivot = df_pivot.sort_values(2025, ascending=False)

# Crear gráfica de barras agrupadas
fig, ax = plt.subplots(figsize=(14, 8))

df_pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Bodega / CEDI', fontsize=12, fontweight='bold')
ax.set_ylabel('Venta Neta (Miles de Millones COP)', fontsize=12, fontweight='bold')
ax.set_title('Comparativo Anual de Ventas por Bodega / CEDI\nTop 10 Bodegas - Duratex Colombia',
             fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
ax.legend(title='Año', loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Gráfica comparativa YoY generada exitosamente")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📋 Resumen de Visualizaciones
# MAGIC 
# MAGIC ### Visualizaciones Generadas
# MAGIC 
# MAGIC 1. ✅ **Tendencia de Ventas Mensual:** Evolución temporal con media móvil y doble eje (ventas + margen)
# MAGIC 2. ✅ **Participación por Línea:** Donut chart con distribución porcentual MDP vs MDF
# MAGIC 3. ✅ **Top 10 Vendedores:** Barras horizontales coloreadas por margen promedio
# MAGIC 4. ✅ **Estacionalidad:** Heatmap mes × año para identificar patrones
# MAGIC 5. ✅ **Ventas vs Devoluciones:** Barras agrupadas con tasa de devolución en eje secundario
# MAGIC 6. ✅ **Márgenes por Sublínea:** Boxplot mostrando distribución y outliers
# MAGIC 7. ✅ **Segmentación RFM:** Scatter interactivo con Plotly para análisis de clientes
# MAGIC 8. ✅ **Comparativo YoY Bodegas:** Barras agrupadas por año para top 10 CEDIs
# MAGIC 
# MAGIC ### Insights Clave
# MAGIC 
# MAGIC * **Tendencias:** Identificación de patrones estacionales y crecimiento por mercado
# MAGIC * **Productos:** MDP y MDF muestran diferentes patrones de margen y volumen
# MAGIC * **Vendedores:** Disparidad en desempeño y tasas de devolución
# MAGIC * **Clientes:** Segmentación RFM revela oportunidades en clientes potenciales y en riesgo
# MAGIC * **Operaciones:** Variabilidad regional en desempeño de bodegas
# MAGIC 
# MAGIC ### Siguientes Pasos
# MAGIC ➡️ Ejecutar `05_modelos_ciencia_datos` para construir modelos predictivos y ML
# MAGIC 
# MAGIC ---
# MAGIC **Proyecto:** Lakehouse Ventas Duratex Colombia  
# MAGIC **Módulo:** Análisis y Visualizaciones  
# MAGIC **Última actualización:** 2026