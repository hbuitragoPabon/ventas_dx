# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Gold Layer Header
# MAGIC %md
# MAGIC # 03 - GOLD: Tablas de Negocio Agregadas
# MAGIC
# MAGIC **Objetivo:** Crear tablas agregadas optimizadas para BI y análisis de negocio desde la capa Silver.
# MAGIC
# MAGIC **Arquitectura:** Medallón Gold Layer
# MAGIC
# MAGIC **Entrada:** `ventas_duratex.silver.ventas_limpias`
# MAGIC
# MAGIC **Salidas:**
# MAGIC * `ventas_duratex.gold.ventas_por_periodo`
# MAGIC * `ventas_duratex.gold.ventas_por_vendedor`
# MAGIC * `ventas_duratex.gold.ventas_por_producto`
# MAGIC * `ventas_duratex.gold.ventas_por_bodega`
# MAGIC * `ventas_duratex.gold.clientes_rfm`

# COMMAND ----------

# DBTITLE 1,Configuration and Read Silver Data
# Configuración inicial
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Crear esquema Gold
spark.sql("CREATE SCHEMA IF NOT EXISTS ventas_duratex.gold")
print("✓ Schema 'ventas_duratex.gold' verificado/creado\n")

# Leer datos de Silver
SOURCE = "ventas_duratex.silver.ventas_limpias"
df = spark.read.table(SOURCE)

print(f"📥 Leyendo desde: {SOURCE}")
print(f"📊 Total registros Silver: {df.count():,}\n")

# Separar ventas y devoluciones para análisis
df_ventas = df.filter(col("TIPO_TRANSACCION") == "VENTA")
df_dev = df.filter(col("IS_DEVOLUCION") == True)

print(f"💰 Ventas: {df_ventas.count():,}")
print(f"🔄 Devoluciones: {df_dev.count():,}")
print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Table 1 Header
# MAGIC %md
# MAGIC ## TABLA 1: Ventas por Período
# MAGIC Agregación temporal de ventas con métricas clave por mes y mercado

# COMMAND ----------

# DBTITLE 1,Create Sales by Period Table
print("📅 CREANDO TABLA: ventas_por_periodo")
print("=" * 80)

# Agrupar por período y mercado
df_periodo = df_ventas.groupBy(
    "FECHA_PERIODO",
    "ANO_PERIODO",
    "MES_PERIODO",
    "TRIMESTRE",
    "Mercado"
).agg(
    sum("VLR_VENTA_NETA").alias("VENTA_NETA_TOTAL"),
    sum("MARGEN_BRUTO").alias("MARGEN_TOTAL"),
    round(avg("PCT_MARGEN"), 2).alias("MARGEN_PROMEDIO_PCT"),
    count("NUMERO_FACTURA").alias("NUM_FACTURAS"),
    sum("CANT_FACTURADA").alias("UNIDADES_TOTAL"),
    countDistinct("ID_CLIENTE").alias("CLIENTES_UNICOS")
).orderBy("FECHA_PERIODO", "Mercado")

# Guardar
df_periodo.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("ventas_duratex.gold.ventas_por_periodo")

print(f"✓ Tabla creada: ventas_duratex.gold.ventas_por_periodo")
print(f"✓ Registros: {df_periodo.count():,}")
print("\nPrimeros 12 meses:")
display(df_periodo.limit(12))

# COMMAND ----------

# DBTITLE 1,Table 2 Header
# MAGIC %md
# MAGIC ## TABLA 2: Ventas por Vendedor
# MAGIC Desempeño de vendedores con métricas de ventas y tasa de devolución

# COMMAND ----------

# DBTITLE 1,Create Sales by Vendor Table
print("👤 CREANDO TABLA: ventas_por_vendedor")
print("=" * 80)

# Agrupar ventas por vendedor y período
df_vendedor = df_ventas.groupBy(
    "NOMBRE_VENDEDOR",
    "ID_VENDEDOR",
    "FECHA_PERIODO",
    "ANO_PERIODO"
).agg(
    sum("VLR_VENTA_NETA").alias("VENTA_NETA_TOTAL"),
    sum("MARGEN_BRUTO").alias("MARGEN_TOTAL"),
    round(avg("PCT_MARGEN"), 2).alias("MARGEN_PROMEDIO_PCT"),
    sum("CANT_FACTURADA").alias("UNIDADES_TOTAL"),
    count("NUMERO_FACTURA").alias("NUM_FACTURAS"),
    countDistinct("ID_CLIENTE").alias("CLIENTES_UNICOS")
)

# Agrupar devoluciones por vendedor y período
df_dev_vendedor = df_dev.groupBy(
    "NOMBRE_VENDEDOR",
    "FECHA_PERIODO"
).agg(
    count("NUMERO_FACTURA").alias("NUM_DEVOLUCIONES"),
    sum("VLR_VENTA_NETA").alias("MONTO_DEVOLUCIONES")
)

# Join ventas con devoluciones
df_vendedor = df_vendedor.join(
    df_dev_vendedor,
    ["NOMBRE_VENDEDOR", "FECHA_PERIODO"],
    "left"
).fillna(0, ["NUM_DEVOLUCIONES", "MONTO_DEVOLUCIONES"])

# Calcular tasa de devolución
df_vendedor = df_vendedor.withColumn(
    "TASA_DEVOLUCION",
    round(
        (col("NUM_DEVOLUCIONES") / (col("NUM_FACTURAS") + col("NUM_DEVOLUCIONES"))) * 100,
        2
    )
)

# Guardar
df_vendedor.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("ventas_duratex.gold.ventas_por_vendedor")

print(f"✓ Tabla creada: ventas_duratex.gold.ventas_por_vendedor")
print(f"✓ Registros: {df_vendedor.count():,}")
print("\nTop 10 vendedores por venta total:")
display(
    df_vendedor.groupBy("NOMBRE_VENDEDOR") \
        .agg(sum("VENTA_NETA_TOTAL").alias("VENTA_TOTAL")) \
        .orderBy(col("VENTA_TOTAL").desc()) \
        .limit(10)
)

# COMMAND ----------

# DBTITLE 1,Table 3 Header
# MAGIC %md
# MAGIC ## TABLA 3: Ventas por Producto
# MAGIC Análisis de productos por línea, sublínea y características

# COMMAND ----------

# DBTITLE 1,Create Sales by Product Table
print("📦 CREANDO TABLA: ventas_por_producto")
print("=" * 80)

# Agrupar por producto
df_producto = df_ventas.groupBy(
    "LINEA",
    "SUBLINEA",
    "CALIBRE",
    "FORMATO",
    "DESCRIPCION_PRODUCTO"
).agg(
    sum("VLR_VENTA_NETA").alias("VENTA_NETA_TOTAL"),
    sum("MARGEN_BRUTO").alias("MARGEN_TOTAL"),
    round(avg("PCT_MARGEN"), 2).alias("MARGEN_PROMEDIO_PCT"),
    sum("CANT_FACTURADA").alias("UNIDADES_TOTAL"),
    countDistinct("ID_CLIENTE").alias("NUM_CLIENTES"),
    count("*").alias("NUM_TRANSACCIONES")
).orderBy(col("VENTA_NETA_TOTAL").desc())

# Guardar
df_producto.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("ventas_duratex.gold.ventas_por_producto")

print(f"✓ Tabla creada: ventas_duratex.gold.ventas_por_producto")
print(f"✓ Registros: {df_producto.count():,}")
print("\nTop 20 productos por venta:")
display(df_producto.limit(20))

# COMMAND ----------

# DBTITLE 1,Table 4 Header
# MAGIC %md
# MAGIC ## TABLA 4: Ventas por Bodega
# MAGIC Análisis de bodegas/CEDIs con comparativo Year-over-Year

# COMMAND ----------

# DBTITLE 1,Create Sales by Warehouse Table
print("🏭 CREANDO TABLA: ventas_por_bodega")
print("=" * 80)

# Agrupar por bodega y período
df_bodega = df_ventas.groupBy(
    "BODEGA",
    "DESCRIPCION_BODEGA",
    "ANO_PERIODO",
    "TRIMESTRE"
).agg(
    sum("VLR_VENTA_NETA").alias("VENTA_NETA_TOTAL"),
    sum("MARGEN_BRUTO").alias("MARGEN_TOTAL"),
    sum("CANT_FACTURADA").alias("UNIDADES_TOTAL"),
    count("NUMERO_FACTURA").alias("NUM_FACTURAS")
).orderBy("BODEGA", "ANO_PERIODO", "TRIMESTRE")

# Añadir comparativo YoY usando Window function
w = Window.partitionBy("BODEGA", "TRIMESTRE").orderBy("ANO_PERIODO")

df_bodega = df_bodega.withColumn(
    "VENTA_ANO_ANTERIOR",
    lag("VENTA_NETA_TOTAL").over(w)
)

df_bodega = df_bodega.withColumn(
    "CRECIMIENTO_YOY_PCT",
    when(
        col("VENTA_ANO_ANTERIOR").isNotNull() & (col("VENTA_ANO_ANTERIOR") != 0),
        round(
            ((col("VENTA_NETA_TOTAL") - col("VENTA_ANO_ANTERIOR")) / col("VENTA_ANO_ANTERIOR")) * 100,
            2
        )
    ).otherwise(None)
)

# Guardar
df_bodega.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("ventas_duratex.gold.ventas_por_bodega")

print(f"✓ Tabla creada: ventas_duratex.gold.ventas_por_bodega")
print(f"✓ Registros: {df_bodega.count():,}")
print("\nMuestra con crecimiento YoY:")
display(df_bodega.filter(col("CRECIMIENTO_YOY_PCT").isNotNull()).limit(10))

# COMMAND ----------

# DBTITLE 1,Table 5 Header
# MAGIC %md
# MAGIC ## TABLA 5: Segmentación RFM de Clientes
# MAGIC Análisis de clientes por Recencia, Frecuencia y Monto (RFM)

# COMMAND ----------

# DBTITLE 1,Create RFM Segmentation Table
print("🎯 CREANDO TABLA: clientes_rfm")
print("=" * 80)

import pandas as pd

# Obtener fecha máxima
fecha_max = df_ventas.agg(max("FECHA_FACTURA")).collect()[0][0]
print(f"Fecha de referencia: {fecha_max}")

# Calcular RFM por cliente
df_rfm = df_ventas.groupBy(
    "ID_CLIENTE",
    "CLIENTE"
).agg(
    datediff(lit(fecha_max), max("FECHA_FACTURA")).alias("RECENCIA"),
    countDistinct("NUMERO_FACTURA").alias("FRECUENCIA"),
    sum("VLR_VENTA_NETA").alias("MONTO")
)

print(f"\n✓ Métricas RFM calculadas para {df_rfm.count():,} clientes")

# Convertir a pandas para calcular scores
df_rfm_pd = df_rfm.toPandas()

print("\nCalculando scores RFM...")

# Calcular scores con qcut (4 cuartiles)
# Recencia: menor es mejor, por eso invertimos
df_rfm_pd["R_score"] = pd.qcut(
    df_rfm_pd["RECENCIA"], 
    q=4, 
    labels=[4, 3, 2, 1],
    duplicates='drop'
)

# Frecuencia: mayor es mejor
df_rfm_pd["F_score"] = pd.qcut(
    df_rfm_pd["FRECUENCIA"].rank(method="first"), 
    q=4, 
    labels=[1, 2, 3, 4],
    duplicates='drop'
)

# Monto: mayor es mejor
df_rfm_pd["M_score"] = pd.qcut(
    df_rfm_pd["MONTO"], 
    q=4, 
    labels=[1, 2, 3, 4],
    duplicates='drop'
)

# Calcular RFM score combinado
df_rfm_pd["RFM_SCORE"] = (
    df_rfm_pd["R_score"].astype(int) + 
    df_rfm_pd["F_score"].astype(int) + 
    df_rfm_pd["M_score"].astype(int)
)

# Segmentar clientes
def segmentar(row):
    r = int(row["R_score"])
    f = int(row["F_score"])
    m = int(row["M_score"])
    
    if r >= 4 and f >= 3:
        return "Champion"
    elif r >= 3 and f >= 3:
        return "Leal"
    elif r >= 3 and f <= 2:
        return "Potencial"
    elif r <= 2 and f >= 3:
        return "En_Riesgo"
    elif r <= 2 and f <= 2:
        return "Inactivo"
    else:
        return "Regular"

df_rfm_pd["SEGMENTO_RFM"] = df_rfm_pd.apply(segmentar, axis=1)

print("✓ Scores y segmentos calculados")

# Convertir de vuelta a Spark
df_rfm_spark = spark.createDataFrame(df_rfm_pd)

# Guardar
df_rfm_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("ventas_duratex.gold.clientes_rfm")

print(f"\n✓ Tabla creada: ventas_duratex.gold.clientes_rfm")
print(f"✓ Registros: {df_rfm_spark.count():,}")

# Distribución de segmentos
print("\nDistribución de clientes por segmento:")
df_segmentos = df_rfm_spark.groupBy("SEGMENTO_RFM") \
    .agg(
        count("*").alias("NUM_CLIENTES"),
        round(sum("MONTO"), 2).alias("VENTA_TOTAL"),
        round(avg("RECENCIA"), 1).alias("RECENCIA_PROM"),
        round(avg("FRECUENCIA"), 1).alias("FRECUENCIA_PROM")
    ) \
    .orderBy(col("VENTA_TOTAL").desc())

display(df_segmentos)

# COMMAND ----------

# DBTITLE 1,Optimization Header
# MAGIC %md
# MAGIC ## Optimización de Tablas Gold
# MAGIC Aplicar Z-ORDER para mejorar el rendimiento de queries

# COMMAND ----------

# DBTITLE 1,Optimize Gold Tables
print("🚀 OPTIMIZANDO TABLAS GOLD")
print("=" * 80)

print("\n1. Optimizando ventas_por_periodo...")
spark.sql("OPTIMIZE ventas_duratex.gold.ventas_por_periodo ZORDER BY (FECHA_PERIODO)")
print("✓ ventas_por_periodo optimizada")

print("\n2. Optimizando ventas_por_vendedor...")
spark.sql("OPTIMIZE ventas_duratex.gold.ventas_por_vendedor ZORDER BY (NOMBRE_VENDEDOR)")
print("✓ ventas_por_vendedor optimizada")

print("\n3. Optimizando ventas_por_producto...")
spark.sql("OPTIMIZE ventas_duratex.gold.ventas_por_producto ZORDER BY (LINEA, SUBLINEA)")
print("✓ ventas_por_producto optimizada")

print("\n4. Optimizando ventas_por_bodega...")
spark.sql("OPTIMIZE ventas_duratex.gold.ventas_por_bodega ZORDER BY (BODEGA)")
print("✓ ventas_por_bodega optimizada")

print("\n5. Optimizando clientes_rfm...")
spark.sql("OPTIMIZE ventas_duratex.gold.clientes_rfm ZORDER BY (SEGMENTO_RFM)")
print("✓ clientes_rfm optimizada")

print("\n" + "=" * 80)
print("✓ TODAS LAS TABLAS OPTIMIZADAS")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Final Summary
print("\n🎆 RESUMEN FINAL - TABLAS GOLD CREADAS")
print("=" * 80)

print("\nTablas disponibles en ventas_duratex.gold:")
spark.sql("SHOW TABLES IN ventas_duratex.gold").show(truncate=False)

# Conteo de registros por tabla
print("\nConteo de registros:")
tablas = [
    "ventas_por_periodo",
    "ventas_por_vendedor",
    "ventas_por_producto",
    "ventas_por_bodega",
    "clientes_rfm"
]

for tabla in tablas:
    count = spark.read.table(f"ventas_duratex.gold.{tabla}").count()
    print(f"  • {tabla}: {count:,} registros")

print("\n" + "=" * 80)
print("✓ PIPELINE GOLD COMPLETADO EXITOSAMENTE")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Documentation
# MAGIC %md
# MAGIC ---
# MAGIC ## 📋 Documentación del Notebook
# MAGIC
# MAGIC ### Descripción
# MAGIC Este notebook implementa la **capa Gold** de la arquitectura Medallón, creando 5 tablas agregadas optimizadas para análisis de negocio, dashboards de BI y modelos de ciencia de datos.
# MAGIC
# MAGIC ### Tabla Origen
# MAGIC * **Tabla:** `ventas_duratex.silver.ventas_limpias`
# MAGIC * **Registros:** 179,075
# MAGIC * **Estado:** Datos limpios y enriquecidos
# MAGIC
# MAGIC ### Tablas Destino
# MAGIC
# MAGIC #### 1. ventas_por_periodo
# MAGIC * **Descripción:** Agregación temporal mensual de ventas por mercado
# MAGIC * **Granularidad:** FECHA_PERIODO + Mercado
# MAGIC * **Métricas:** Venta neta, margen, facturas, unidades, clientes únicos
# MAGIC * **Uso:** Análisis de tendencias, estacionalidad, forecasting
# MAGIC
# MAGIC #### 2. ventas_por_vendedor
# MAGIC * **Descripción:** Desempeño de vendedores con devoluciones
# MAGIC * **Granularidad:** NOMBRE_VENDEDOR + FECHA_PERIODO
# MAGIC * **Métricas:** Ventas, márgenes, devoluciones, tasa de devolución
# MAGIC * **Uso:** Evaluación de desempeño, comisiones, gestión comercial
# MAGIC
# MAGIC #### 3. ventas_por_producto
# MAGIC * **Descripción:** Análisis de productos por características
# MAGIC * **Granularidad:** LINEA + SUBLINEA + CALIBRE + FORMATO
# MAGIC * **Métricas:** Ventas, márgenes, unidades, clientes, transacciones
# MAGIC * **Uso:** Portfolio de productos, pricing, mix de ventas
# MAGIC
# MAGIC #### 4. ventas_por_bodega
# MAGIC * **Descripción:** Desempeño de bodegas/CEDIs con YoY
# MAGIC * **Granularidad:** BODEGA + ANO_PERIODO + TRIMESTRE
# MAGIC * **Métricas:** Ventas, márgenes, unidades, crecimiento YoY
# MAGIC * **Uso:** Operaciones logísticas, planificación de inventario
# MAGIC
# MAGIC #### 5. clientes_rfm
# MAGIC * **Descripción:** Segmentación de clientes por RFM
# MAGIC * **Granularidad:** ID_CLIENTE
# MAGIC * **Métricas:** Recencia, Frecuencia, Monto, scores, segmento
# MAGIC * **Segmentos:** Champion, Leal, Potencial, En_Riesgo, Inactivo, Regular
# MAGIC * **Uso:** CRM, campañas de marketing, retención de clientes
# MAGIC
# MAGIC ### Optimizaciones Aplicadas
# MAGIC * **Z-ORDER:** Aplicado en columnas clave de cada tabla para mejorar performance
# MAGIC * **Delta Lake:** Todas las tablas en formato Delta optimizado
# MAGIC * **Particionamiento:** Según necesidades de acceso de cada tabla
# MAGIC
# MAGIC ### Métricas Calculadas
# MAGIC | Métrica | Fórmula | Descripción |
# MAGIC |----------|---------|-------------|
# MAGIC | VENTA_NETA_TOTAL | SUM(VLR_VENTA_NETA) | Total de ventas netas |
# MAGIC | MARGEN_TOTAL | SUM(MARGEN_BRUTO) | Margen bruto total |
# MAGIC | MARGEN_PROMEDIO_PCT | AVG(PCT_MARGEN) | Porcentaje de margen promedio |
# MAGIC | TASA_DEVOLUCION | (Devoluciones / Total) × 100 | % de transacciones devueltas |
# MAGIC | CRECIMIENTO_YOY_PCT | ((Año actual - Año anterior) / Año anterior) × 100 | Crecimiento interanual |
# MAGIC | RFM_SCORE | R_score + F_score + M_score | Puntuación combinada RFM (3-12) |
# MAGIC
# MAGIC ### Siguientes Pasos
# MAGIC ➡️ Ejecutar `04_analisis_visualizaciones` para generar gráficas y dashboards
# MAGIC
# MAGIC ➡️ Ejecutar `05_modelos_ciencia_datos` para entrenar modelos predictivos
# MAGIC
# MAGIC ---
# MAGIC **Proyecto:** Lakehouse Ventas Duratex Colombia  
# MAGIC **Capa:** Gold (Datos Agregados de Negocio)  
# MAGIC **Última actualización:** 2026
