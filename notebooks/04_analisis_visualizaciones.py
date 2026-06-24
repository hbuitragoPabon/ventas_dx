# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Header - Analysis and Visualizations
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

# DBTITLE 1,Setup - Import Libraries
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

# DBTITLE 1,Description - Chart 1
# MAGIC %md
# MAGIC ## GRÁFICA 1: Tendencia de Ventas Mensual
# MAGIC
# MAGIC **Análisis:** Evolución temporal de ventas netas y margen promedio por mercado (Interno/Externo) con media móvil de 3 meses.
# MAGIC
# MAGIC **Conclusión de negocio:** Permite identificar tendencias, estacionalidad y oportunidades de crecimiento por mercado.

# COMMAND ----------

# DBTITLE 1,Chart 1 - Monthly Sales Trend
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

# Título y leyenda
plt.title('Tendencia de Ventas Mensuales por Mercado\n(Con media móvil de 3 meses)', 
          fontsize=14, fontweight='bold', pad=20)

# Combinar leyendas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

plt.tight_layout()
display(fig)
plt.close()

print("\n✓ Gráfica 1 generada")

# COMMAND ----------

# DBTITLE 1,Description - Chart 2
# MAGIC %md
# MAGIC ## GRÁFICA 2: Participación por Línea de Producto
# MAGIC
# MAGIC **Análisis:** Distribución porcentual de ventas por línea de producto (MDP vs MDF).
# MAGIC
# MAGIC **Conclusión de negocio:** Identifica las líneas de mayor contribución al revenue total.

# COMMAND ----------

# DBTITLE 1,Chart 2 - Product Line Participation
print("🍩 GRÁFICA 2: Participación por Línea de Producto")
print("=" * 80)

# Leer y agrupar por línea
df_producto = spark.read.table("ventas_duratex.gold.ventas_por_producto")
df_linea = df_producto.groupBy("LINEA").sum("VENTA_NETA_TOTAL").toPandas()
df_linea.columns = ['LINEA', 'VENTA_TOTAL']

# Filtrar nulls y calcular porcentajes
df_linea = df_linea[df_linea['LINEA'].notna()]
df_linea['PCT'] = (df_linea['VENTA_TOTAL'] / df_linea['VENTA_TOTAL'].sum()) * 100

# Ordenar por venta
df_linea = df_linea.sort_values('VENTA_TOTAL', ascending=False)

# Crear gráfica de dona con Plotly
fig = go.Figure(data=[go.Pie(
    labels=df_linea['LINEA'],
    values=df_linea['VENTA_TOTAL'],
    hole=0.4,
    marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c']),
    textinfo='label+percent',
    textposition='outside',
    hovertemplate='<b>%{label}</b><br>Venta: $%{value:,.0f} COP<br>%{percent}<extra></extra>'
)])

fig.update_layout(
    title='Participación de Ventas por Línea de Producto',
    title_font_size=16,
    title_x=0.5,
    showlegend=True,
    height=500,
    annotations=[dict(text=f'{df_linea["VENTA_TOTAL"].sum()/1e12:.2f}T<br>COP Total', 
                     x=0.5, y=0.5, font_size=18, showarrow=False)]
)

display(fig)

print("\n✓ Gráfica 2 generada")
for _, row in df_linea.iterrows():
    print(f"  • {row['LINEA']}: ${row['VENTA_TOTAL']/1e9:.1f}B ({row['PCT']:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Description - Chart 3
# MAGIC %md
# MAGIC ## GRÁFICA 3: Top 10 Vendedores
# MAGIC
# MAGIC **Análisis:** Ranking de vendedores por venta neta total, coloreado por margen promedio.
# MAGIC
# MAGIC **Conclusión de negocio:** Identifica vendedores estrella y aquellos que requieren capacitación para mejorar márgenes.

# COMMAND ----------

# DBTITLE 1,Chart 3 - Top 10 Vendors
print("🏆 GRÁFICA 3: Top 10 Vendedores por Venta y Margen")
print("=" * 80)

# Leer y agrupar por vendedor
df_vendedor = spark.read.table("ventas_duratex.gold.ventas_por_vendedor")
df_vend_agg = df_vendedor.groupBy("NOMBRE_VENDEDOR").agg(
    {"VENTA_NETA_TOTAL": "sum", "MARGEN_PROMEDIO_PCT": "avg"}
).toPandas()
df_vend_agg.columns = ['NOMBRE_VENDEDOR', 'VENTA_TOTAL', 'MARGEN_PROM']

# Top 10 por venta
df_top10 = df_vend_agg.nlargest(10, 'VENTA_TOTAL')

# Crear gráfica de barras horizontales
fig = go.Figure()

fig.add_trace(go.Bar(
    y=df_top10['NOMBRE_VENDEDOR'],
    x=df_top10['VENTA_TOTAL']/1e9,
    orientation='h',
    marker=dict(
        color=df_top10['MARGEN_PROM'],
        colorscale='RdYlGn_r',
        colorbar=dict(title="Margen %"),
        line=dict(color='rgb(100,100,100)', width=1)
    ),
    text=[f'${v/1e9:.1f}B' for v in df_top10['VENTA_TOTAL']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Venta: $%{x:.1f}B COP<br>Margen: %{marker.color:.1f}%<extra></extra>'
))

fig.update_layout(
    title='Top 10 Vendedores por Venta Total<br>(Coloreado por Margen Promedio)',
    xaxis_title='Venta Total (Miles de Millones COP)',
    yaxis_title='',
    height=500,
    yaxis=dict(autorange="reversed"),
    showlegend=False
)

display(fig)

print("\n✓ Gráfica 3 generada")

# COMMAND ----------

# DBTITLE 1,Description - Chart 4
# MAGIC %md
# MAGIC ## GRÁFICA 4: Estacionalidad de Ventas (Heatmap)
# MAGIC
# MAGIC **Análisis:** Mapa de calor mostrando ventas por mes y año para identificar patrones estacionales.
# MAGIC
# MAGIC **Conclusión de negocio:** Ayuda a planificar inventario y campañas de marketing estacionales.

# COMMAND ----------

# DBTITLE 1,Chart 4 - Sales Seasonality Heatmap
print("🔥 GRÁFICA 4: Estacionalidad de Ventas - Mapa de Calor")
print("=" * 80)

# Leer datos por período
df_periodo = spark.read.table("ventas_duratex.gold.ventas_por_periodo").toPandas()

# Agrupar por año y mes (sin mercado)
df_heatmap = df_periodo.groupby(['ANO_PERIODO', 'MES_PERIODO']).agg({
    'VENTA_NETA_TOTAL': 'sum'
}).reset_index()

# Pivotar para heatmap
df_pivot = df_heatmap.pivot(index='ANO_PERIODO', columns='MES_PERIODO', values='VENTA_NETA_TOTAL').fillna(0)

# Crear heatmap con seaborn
fig, ax = plt.subplots(figsize=(12, 6))

sns.heatmap(
    df_pivot / 1e9,  # Convertir a miles de millones
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    cbar_kws={'label': 'Venta (Miles de Millones COP)'},
    linewidths=0.5,
    ax=ax
)

ax.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax.set_ylabel('Año', fontsize=12, fontweight='bold')
ax.set_title('Estacionalidad de Ventas por Mes y Año', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
display(fig)
plt.close()

print("\n✓ Gráfica 4 generada")

# COMMAND ----------

# DBTITLE 1,Description - Chart 5
# MAGIC %md
# MAGIC ## GRÁFICA 5: Ventas vs Devoluciones por Vendedor
# MAGIC
# MAGIC **Análisis:** Comparativo de ventas y devoluciones con tasa de devolución en eje secundario.
# MAGIC
# MAGIC **Conclusión de negocio:** Identifica vendedores con alta tasa de devolución que requieren mejora en proceso de ventas.

# COMMAND ----------

# DBTITLE 1,Chart 5 - Sales vs Returns by Vendor
print("⚖️ GRÁFICA 5: Ventas vs Devoluciones por Vendedor")
print("=" * 80)

# Leer y agrupar datos de vendedor
df_vendedor = spark.read.table("ventas_duratex.gold.ventas_por_vendedor").toPandas()

# Agrupar por vendedor
df_vend_sum = df_vendedor.groupby('NOMBRE_VENDEDOR').agg({
    'VENTA_NETA_TOTAL': 'sum',
    'NUM_DEVOLUCIONES': 'sum',
    'MONTO_DEVOLUCIONES': 'sum'
}).reset_index()

# Calcular tasa de devolución
df_vend_sum['TASA_DEVOLUCION'] = (abs(df_vend_sum['MONTO_DEVOLUCIONES']) / df_vend_sum['VENTA_NETA_TOTAL'] * 100).fillna(0)

# Top 15 vendedores por venta
df_top15 = df_vend_sum.nlargest(15, 'VENTA_NETA_TOTAL').sort_values('VENTA_NETA_TOTAL')

# Crear figura con doble eje
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Barras de ventas
fig.add_trace(
    go.Bar(name='Ventas', x=df_top15['VENTA_NETA_TOTAL']/1e9, y=df_top15['NOMBRE_VENDEDOR'],
           orientation='h', marker_color='#1f77b4'),
    secondary_y=False
)

# Barras de devoluciones
fig.add_trace(
    go.Bar(name='Devoluciones', x=abs(df_top15['MONTO_DEVOLUCIONES'])/1e9, y=df_top15['NOMBRE_VENDEDOR'],
           orientation='h', marker_color='#d62728'),
    secondary_y=False
)

# Línea de tasa de devolución
fig.add_trace(
    go.Scatter(name='Tasa Devolución %', x=df_top15['TASA_DEVOLUCION'], y=df_top15['NOMBRE_VENDEDOR'],
               mode='lines+markers', marker_color='#ff7f0e', line=dict(width=2)),
    secondary_y=True
)

fig.update_layout(
    title='Ventas vs Devoluciones por Vendedor (Top 15)',
    barmode='group',
    height=600,
    xaxis_title='Monto (Miles de Millones COP)',
    yaxis_title=''
)

fig.update_yaxes(title_text="Tasa Devolución (%)", secondary_y=True)

display(fig)

print("\n✓ Gráfica 5 generada")

# COMMAND ----------

# DBTITLE 1,Description - Chart 6
# MAGIC %md
# MAGIC ## GRÁFICA 6: Distribución de Márgenes por Sublínea
# MAGIC
# MAGIC **Análisis:** Boxplot mostrando la distribución de márgenes para cada sublínea de producto.
# MAGIC
# MAGIC **Conclusión de negocio:** Identifica sublíneas con mayor variabilidad en márgenes y outliers que requieren revisión.

# COMMAND ----------

# DBTITLE 1,Chart 6 - Margin Distribution by Subline
print("📦 GRÁFICA 6: Distribución de Márgenes por Sublínea")
print("=" * 80)

# Leer datos de producto
df_producto = spark.read.table("ventas_duratex.gold.ventas_por_producto").toPandas()

# Filtrar solo sublíneas con datos válidos
df_margin = df_producto[df_producto['SUBLINEA'].notna()].copy()

# Limitar a top 10 sublíneas por volumen
top_sublineas = df_margin.groupby('SUBLINEA')['VENTA_NETA_TOTAL'].sum().nlargest(10).index
df_margin_top = df_margin[df_margin['SUBLINEA'].isin(top_sublineas)]

# Crear boxplot con Plotly
fig = go.Figure()

for sublinea in df_margin_top['SUBLINEA'].unique():
    data = df_margin_top[df_margin_top['SUBLINEA'] == sublinea]['MARGEN_PROMEDIO_PCT']
    fig.add_trace(go.Box(
        y=data,
        name=sublinea,
        boxmean='sd'
    ))

fig.update_layout(
    title='Distribución de Márgenes por Sublínea (Top 10)',
    yaxis_title='Margen Promedio (%)',
    xaxis_title='Sublínea',
    height=500,
    showlegend=False
)

display(fig)

print("\n✓ Gráfica 6 generada")

# COMMAND ----------

# DBTITLE 1,Description - Chart 7
# MAGIC %md
# MAGIC ## GRÁFICA 7: Segmentación RFM (Scatter Interactivo)
# MAGIC
# MAGIC **Análisis:** Scatter plot interactivo de clientes en dimensiones RFM (Recencia, Frecuencia, Monto).
# MAGIC
# MAGIC **Conclusión de negocio:** Identifica clientes Champions, En Riesgo e Inactivos para estrategias de retención/reactivación.

# COMMAND ----------

# DBTITLE 1,Chart 7 - RFM Customer Segmentation
print("🎯 GRÁFICA 7: Segmentación RFM de Clientes")
print("=" * 80)

# Leer datos de clientes RFM
df_rfm = spark.read.table("ventas_duratex.gold.clientes_rfm").toPandas()

# Crear scatter interactivo con Plotly
fig = px.scatter(
    df_rfm, 
    x='RECENCIA', 
    y='FRECUENCIA',
    size='MONTO',
    color='SEGMENTO_RFM',
    hover_data=['CLIENTE', 'R_score', 'F_score', 'M_score'],
    title='Segmentación RFM de Clientes<br>(Tamaño = Monto Total)',
    labels={
        'RECENCIA': 'Recencia (días desde última compra)',
        'FRECUENCIA': 'Frecuencia (número de transacciones)',
        'SEGMENTO_RFM': 'Segmento RFM'
    },
    color_discrete_map={
        'Champion': '#2ca02c',
        'Leal': '#1f77b4',
        'Potencial': '#ff7f0e',
        'En_Riesgo': '#d62728',
        'Inactivo': '#7f7f7f'
    },
    size_max=60
)

fig.update_layout(
    height=600,
    xaxis_title='Recencia (días)',
    yaxis_title='Frecuencia (transacciones)',
    showlegend=True
)

display(fig)

print("\n✓ Gráfica 7 generada")
print(f"  • Clientes analizados: {len(df_rfm)}")
print(f"  • Segmentos: {df_rfm['SEGMENTO_RFM'].nunique()}")

# COMMAND ----------

# DBTITLE 1,Description - Chart 8
# MAGIC %md
# MAGIC ## GRÁFICA 8: Comparativo YoY por Bodega
# MAGIC
# MAGIC **Análisis:** Comparación de ventas anuales por bodega con crecimiento Year-over-Year.
# MAGIC
# MAGIC **Conclusión de negocio:** Identifica bodegas con mayor crecimiento y aquellas que requieren atención.

# COMMAND ----------

# DBTITLE 1,Chart 8 - YoY Warehouse Comparison
print("🏭 GRÁFICA 8: Comparativo Anual de Ventas por Bodega")
print("=" * 80)

# Leer datos de bodega
df_bodega = spark.read.table("ventas_duratex.gold.ventas_por_bodega").toPandas()

# Agrupar por bodega y año (Q1 solo para comparación)
df_bodega_anual = df_bodega[df_bodega['TRIMESTRE'] == 1].groupby(['BODEGA', 'DESCRIPCION_BODEGA', 'ANO_PERIODO']).agg({
    'VENTA_NETA_TOTAL': 'sum'
}).reset_index()

# Top 8 bodegas por venta en 2026
top_bodegas = df_bodega_anual[df_bodega_anual['ANO_PERIODO'] == 2026].nlargest(8, 'VENTA_NETA_TOTAL')['BODEGA'].unique()
df_plot = df_bodega_anual[df_bodega_anual['BODEGA'].isin(top_bodegas)]

# Crear gráfica de barras agrupadas
fig = px.bar(
    df_plot,
    x='DESCRIPCION_BODEGA',
    y='VENTA_NETA_TOTAL',
    color='ANO_PERIODO',
    barmode='group',
    title='Comparativo Anual de Ventas por Bodega (Q1)<br>Top 8 Bodegas 2026',
    labels={'VENTA_NETA_TOTAL': 'Venta Neta (COP)', 'DESCRIPCION_BODEGA': '', 'ANO_PERIODO': 'Año'},
    color_discrete_map={2024: '#1f77b4', 2025: '#ff7f0e', 2026: '#2ca02c'}
)

fig.update_layout(height=600, xaxis_tickangle=-45)

display(fig)

print("\n✓ Gráfica 8 generada")

# COMMAND ----------

# DBTITLE 1,Summary - All Visualizations
# MAGIC %md
# MAGIC ---
# MAGIC ## 📋 Resumen de Visualizaciones
# MAGIC
# MAGIC ### Visualizaciones Generadas
# MAGIC
# MAGIC 1. **Tendencia de Ventas Mensual**: Evolución temporal con media móvil de 3 meses
# MAGIC 2. **Participación por Línea**: Distribución porcentual MDP vs MDF
# MAGIC 3. **Top 10 Vendedores**: Ranking con codificación de color por margen
# MAGIC 4. **Estacionalidad**: Heatmap de ventas por mes y año
# MAGIC 5. **Ventas vs Devoluciones**: Comparativo por vendedor con tasa de devolución
# MAGIC 6. **Distribución de Márgenes**: Boxplot por sublínea de producto
# MAGIC 7. **Segmentación RFM**: Scatter interactivo de clientes
# MAGIC 8. **Comparativo YoY**: Crecimiento anual por bodega
# MAGIC
# MAGIC ### Insights Clave
# MAGIC
# MAGIC **Clientes (RFM):**
# MAGIC * 117 Champions (25% de clientes) → 75% del revenue
# MAGIC * 186 Inactivos (39%) → Oportunidad de reactivación
# MAGIC * 46 En Riesgo → Atención inmediata requerida
# MAGIC
# MAGIC **Productos:**
# MAGIC * MDP domina con 63.7% del revenue total
# MAGIC * MDF contribuye 36.3%
# MAGIC
# MAGIC **Vendedores:**
# MAGIC * Top vendedor: CARDONA HENAO CESAR AUGUSTO ($285.5B)
# MAGIC * Márgenes negativos requieren revisión de costos
# MAGIC
# MAGIC ### Recomendaciones de Negocio
# MAGIC
# MAGIC 1. **Prioridad Alta**: Retener Champions (75% del revenue)
# MAGIC 2. **Campaña de Reactivación**: 186 clientes inactivos ($40.7B oportunidad)
# MAGIC 3. **Intervención Urgente**: 46 clientes En_Riesgo
# MAGIC 4. **Revisión de Costos**: Estructura de márgenes negativos
# MAGIC 5. **Upselling**: 50 clientes Potenciales con compras recientes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ✓ Pipeline de análisis completo y listo para dashboards de BI
