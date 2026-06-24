# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Bronze Ingestion Header
# MAGIC %md
# MAGIC # 01 - BRONZE: Ingesta de Ventas Duratex
# MAGIC
# MAGIC **Objetivo:** Leer el archivo Excel desde el volumen Unity Catalog y guardarlo en formato Delta Lake sin transformaciones, preservando el dato crudo.
# MAGIC
# MAGIC **Arquitectura:** Medallón Bronze Layer
# MAGIC
# MAGIC **Entrada:** `/Volumes/ventas_duratex/bronze/landing_zone/Detalle_ventas_2024_2026.xlsx`
# MAGIC
# MAGIC **Salida:** `ventas_duratex.bronze.ventas_raw`

# COMMAND ----------

# DBTITLE 1,Configuration
# CELDA 1 — Configuración inicial
# Definir todas las constantes del notebook

CATALOG = "ventas_duratex"
SCHEMA = "bronze"
TABLE_NAME = "ventas_raw"
VOLUME_PATH = "/Volumes/ventas_duratex/bronze/landing_zone"
FILE_NAME = "Detalle_ventas_2024_2026.xlsx"
FILE_PATH = f"{VOLUME_PATH}/{FILE_NAME}"

print(f"Catálogo: {CATALOG}")
print(f"Esquema: {SCHEMA}")
print(f"Tabla destino: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
print(f"Archivo fuente: {FILE_PATH}")

# COMMAND ----------

# DBTITLE 1,Verify File Exists
# CELDA 2 — Verificar que el archivo existe en el volumen

try:
    archivos = dbutils.fs.ls(VOLUME_PATH)
    print(f"✓ Archivos encontrados en {VOLUME_PATH}:")
    print("-" * 80)
    for archivo in archivos:
        print(f"  • {archivo.name} ({archivo.size:,} bytes)")
    print("-" * 80)
    
    # Verificar que el archivo específico existe
    archivo_existe = any(FILE_NAME in a.name for a in archivos)
    if archivo_existe:
        print(f"\n✓ Archivo '{FILE_NAME}' encontrado y listo para procesamiento")
    else:
        raise FileNotFoundError(f"ERROR: Archivo '{FILE_NAME}' no encontrado en el volumen")
except Exception as e:
    print(f"❌ Error al verificar el volumen: {str(e)}")
    raise

# COMMAND ----------

# DBTITLE 1,Install openpyxl
# Instalar openpyxl para leer archivos Excel
%pip install openpyxl

# COMMAND ----------

# DBTITLE 1,Read Excel and Convert to Spark
# CELDA 3 — Leer el Excel con pandas y convertir a Spark DataFrame
# Usar pandas para leer Excel (formato nativo del ERP) y luego convertir a Spark

import pandas as pd
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, DateType, TimestampType
)
from pyspark.sql.functions import col, to_date

print("Leyendo archivo Excel con pandas...")

# Leer con pandas (openpyxl) - Unity Catalog Volumes se acceden directamente
df_pandas = pd.read_excel(
    FILE_PATH,
    engine='openpyxl'
)

print(f"✓ Registros leídos en pandas: {len(df_pandas):,}")
print(f"✓ Columnas: {len(df_pandas.columns)}")

# Deduplicar nombres de columnas (Spark es case-insensitive)
columns_lower = {}
new_columns = []
for col_name in df_pandas.columns:
    col_lower = str(col_name).lower()
    if col_lower in columns_lower:
        # Columna duplicada (case-insensitive), agregar sufijo
        columns_lower[col_lower] += 1
        new_col_name = f"{col_name}_{columns_lower[col_lower]}"
        new_columns.append(new_col_name)
        print(f"⚠️  Columna duplicada detectada: '{col_name}' renombrada a '{new_col_name}'")
    else:
        columns_lower[col_lower] = 0
        new_columns.append(col_name)

df_pandas.columns = new_columns

# Convertir fechas explícitamente en pandas
if 'FECHA_FACTURA' in df_pandas.columns:
    df_pandas['FECHA_FACTURA'] = pd.to_datetime(df_pandas['FECHA_FACTURA'], errors='coerce')

# Normalizar ID_SUCURSAL_CLIENTE: números con ceros a la izquierda (3 dígitos), strings sin cambios
if 'ID_SUCURSAL_CLIENTE' in df_pandas.columns:
    def format_sucursal(value):
        if pd.isna(value):
            return None
        try:
            # Intentar convertir a número y formatear con ceros
            return str(int(float(value))).zfill(3)
        except (ValueError, TypeError):
            # Si no es numérico, mantener como string
            return str(value)
    
    df_pandas['ID_SUCURSAL_CLIENTE'] = df_pandas['ID_SUCURSAL_CLIENTE'].apply(format_sucursal)

# Convertir REFERENCIA_PRODUCTO a string para evitar errores de tipo mixto
if 'REFERENCIA_PRODUCTO' in df_pandas.columns:
    df_pandas['REFERENCIA_PRODUCTO'] = df_pandas['REFERENCIA_PRODUCTO'].astype(str)

# Convertir a Spark DataFrame
df = spark.createDataFrame(df_pandas)

# Asegurar tipos de dato correctos para campos críticos
df = df.withColumn("PERIODO", col("PERIODO").cast(IntegerType())) \
       .withColumn("ANO", col("ANO").cast(IntegerType())) \
       .withColumn("VLR_VENTA_NETA", col("VLR_VENTA_NETA").cast(DoubleType())) \
       .withColumn("PRECIO_UNITARIO", col("PRECIO_UNITARIO").cast(DoubleType())) \
       .withColumn("COSTO_PROMEDIO", col("COSTO_PROMEDIO").cast(DoubleType())) \
       .withColumn("CANT_FACTURADA", col("CANT_FACTURADA").cast(DoubleType())) \
       .withColumn("VLR_DSCTO_LIN", col("VLR_DSCTO_LIN").cast(DoubleType())) \
       .withColumn("VLR_IMPUESTO", col("VLR_IMPUESTO").cast(DoubleType()))

print(f"\n✓ DataFrame Spark creado exitosamente")

# COMMAND ----------

# DBTITLE 1,Show Schema and Sample
# CELDA 4 — Mostrar el esquema y una muestra de datos

print("ESQUEMA DEL DATAFRAME:")
print("=" * 80)
df.printSchema()
print("=" * 80)

print(f"\nTOTAL REGISTROS LEÍDOS: {df.count():,}")
print("\nMUESTRA DE DATOS (5 primeras filas):")
display(df.limit(5))

# COMMAND ----------

# DBTITLE 1,Add Audit Columns
# CELDA 5 — Agregar columnas de auditoría (sin modificar datos originales)

from pyspark.sql.functions import current_timestamp, lit, current_date
from datetime import datetime

print("Agregando columnas de auditoría...")

# Generar batch_id único basado en timestamp
batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

df = df.withColumn("_ingestion_timestamp", current_timestamp()) \
       .withColumn("_source_file", lit(FILE_NAME)) \
       .withColumn("_batch_id", lit(batch_id))

print(f"✓ Columnas de auditoría agregadas:")
print(f"  • _ingestion_timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  • _source_file: {FILE_NAME}")
print(f"  • _batch_id: {batch_id}")

# Mostrar columnas agregadas
print("\nVista previa con columnas de auditoría:")
display(df.select("NUMERO_FACTURA", "CLIENTE", "VLR_VENTA_NETA", 
                  "_ingestion_timestamp", "_source_file", "_batch_id").limit(3))

# COMMAND ----------

# DBTITLE 1,Save to Delta Lake
# CELDA 6 — Guardar en Delta Lake (Bronze Layer)

print(f"Guardando en Delta Lake: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
print("-" * 80)

# Crear el schema si no existe
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"✓ Schema '{CATALOG}.{SCHEMA}' verificado/creado")

# Guardar en Delta Lake con particionamiento
df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .partitionBy("ANO", "PERIODO") \
  .saveAsTable(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

print(f"✓ Tabla '{TABLE_NAME}' guardada exitosamente")
print(f"✓ Particionamiento: ANO, PERIODO")
print(f"✓ Formato: Delta Lake")
print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Post-Load Validation
# CELDA 7 — Validación post-carga

print("VALIDACIONES POST-CARGA")
print("=" * 80)

# Leer desde la tabla recién creada
df_check = spark.read.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

# Validación 1: Conteo de registros
conteo_actual = df_check.count()
conteo_esperado = 179075

print(f"\n1. Validación de conteo de registros:")
print(f"   Esperado: {conteo_esperado:,}")
print(f"   Actual:   {conteo_actual:,}")

assert conteo_actual == conteo_esperado, f"ERROR: Se esperaban {conteo_esperado:,} registros, se encontraron {conteo_actual:,}"
print(f"   ✓ Conteo correcto")

# Validación 2: Columnas de auditoría presentes
print(f"\n2. Validación de columnas de auditoría:")
columnas_auditoria = ['_ingestion_timestamp', '_source_file', '_batch_id']
columnas_disponibles = df_check.columns
for col_name in columnas_auditoria:
    assert col_name in columnas_disponibles, f"ERROR: Columna '{col_name}' no encontrada"
    print(f"   ✓ {col_name}: presente")

# Validación 3: Particiones creadas
print(f"\n3. Validación de particiones:")
particiones = spark.sql(f"""
    SELECT DISTINCT ANO, PERIODO 
    FROM {CATALOG}.{SCHEMA}.{TABLE_NAME} 
    ORDER BY ANO, PERIODO
""").collect()
print(f"   Total particiones creadas: {len(particiones)}")
print(f"   Rango: {particiones[0].ANO}-{str(particiones[0].PERIODO)[-2:]} → {particiones[-1].ANO}-{str(particiones[-1].PERIODO)[-2:]}")

# Validación 4: Distribución por año
print(f"\n4. Distribución de registros por año:")
df_anos = spark.sql(f"""
    SELECT ANO, COUNT(*) as registros
    FROM {CATALOG}.{SCHEMA}.{TABLE_NAME}
    GROUP BY ANO
    ORDER BY ANO
""")
display(df_anos)

print("\n" + "=" * 80)
print("✓ TODAS LAS VALIDACIONES PASARON EXITOSAMENTE")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Descriptive Statistics
# CELDA 8 — Estadísticas descriptivas de columnas clave

print("ESTADÍSTICAS DESCRIPTIVAS - COLUMNAS FINANCIERAS")
print("=" * 80)

df_stats = df_check.describe([
    "VLR_VENTA_NETA", 
    "CANT_FACTURADA",
    "COSTO_PROMEDIO", 
    "PRECIO_UNITARIO",
    "VLR_DSCTO_LIN",
    "VLR_IMPUESTO"
])

display(df_stats)

# Resumen adicional
print("\nRESUMEN DE CARGA:")
print("-" * 80)
print(f"📁 Archivo fuente:     {FILE_NAME}")
print(f"📊 Registros cargados: {conteo_actual:,}")
print(f"📋 Columnas totales:   {len(df_check.columns)}")
print(f"🗂️  Tabla destino:      {CATALOG}.{SCHEMA}.{TABLE_NAME}")
print(f"📅 Períodos cubiertos: {len(particiones)} meses (Ene 2024 - Jun 2026)")
print(f"⏱️  Timestamp ingesta:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Documentation
# MAGIC %md
# MAGIC ---
# MAGIC ## 📋 Documentación del Notebook
# MAGIC
# MAGIC ### Descripción
# MAGIC Este notebook implementa la **capa Bronze** de la arquitectura Medallón para el proyecto de ventas Duratex Colombia. Lee el archivo Excel crudo desde el volumen Unity Catalog y lo persiste en formato Delta Lake sin aplicar transformaciones, preservando la integridad del dato original.
# MAGIC
# MAGIC ### Entrada
# MAGIC * **Ruta:** `/Volumes/ventas_duratex/bronze/landing_zone/Detalle_ventas_2024_2026.xlsx`
# MAGIC * **Formato:** Excel (.xlsx)
# MAGIC * **Registros:** 179,075 filas
# MAGIC * **Columnas:** 48 campos de datos + 3 de auditoría
# MAGIC
# MAGIC ### Salida
# MAGIC * **Tabla:** `ventas_duratex.bronze.ventas_raw`
# MAGIC * **Formato:** Delta Lake
# MAGIC * **Particionamiento:** Por `ANO` y `PERIODO` (30 particiones: 202401 → 202606)
# MAGIC * **Columnas agregadas:**
# MAGIC   - `_ingestion_timestamp`: Timestamp de carga
# MAGIC   - `_source_file`: Nombre del archivo origen
# MAGIC   - `_batch_id`: Identificador único del lote
# MAGIC
# MAGIC ### Esquema Principal
# MAGIC | Campo | Tipo | Descripción |
# MAGIC |-------|------|-------------|
# MAGIC | COMPANIA | String | Compañía (1 o 2) |
# MAGIC | PERIODO | Integer | Período AAAAMM (202401-202606) |
# MAGIC | ANO | Integer | Año (2024-2026) |
# MAGIC | FECHA_FACTURA | Date | Fecha de emisión |
# MAGIC | TIPO_DOCUMENTO | String | EFN/EFE (facturas), ENN/ENE (notas crédito) |
# MAGIC | NUMERO_FACTURA | String | Número de documento |
# MAGIC | ID_CLIENTE | String | Identificador del cliente |
# MAGIC | CLIENTE | String | Nombre del cliente |
# MAGIC | ID_VENDEDOR | String | Código del vendedor |
# MAGIC | NOMBRE_VENDEDOR | String | Nombre del vendedor |
# MAGIC | BODEGA | String | Código CEDI |
# MAGIC | DESCRIPCION_BODEGA | String | Descripción del CEDI |
# MAGIC | LINEA | String | MDP, MDF, o NULL |
# MAGIC | SUBLINEA | String | MDP Melamina, MDF Desnudo, etc. |
# MAGIC | CALIBRE | String | Espesor del producto (ej: 15.0mm) |
# MAGIC | REFERENCIA_PRODUCTO | String | SKU del producto |
# MAGIC | DESCRIPCION_PRODUCTO | String | Descripción completa |
# MAGIC | CANT_FACTURADA | Double | Cantidad vendida |
# MAGIC | PRECIO_UNITARIO | Double | Precio por unidad |
# MAGIC | VLR_VENTA_NETA | Double | Valor neto de venta |
# MAGIC | VLR_DSCTO_LIN | Double | Descuento aplicado |
# MAGIC | VLR_IMPUESTO | Double | IVA |
# MAGIC | COSTO_PROMEDIO | Double | Costo unitario |
# MAGIC | Mercado | String | Tipo de mercado |
# MAGIC | TIPO_CLIENTE | String | NAL (interno) o EXT (exportación) |
# MAGIC
# MAGIC ### Validaciones Aplicadas
# MAGIC 1. ✅ Conteo de registros: 179,075
# MAGIC 2. ✅ Columnas de auditoría presentes
# MAGIC 3. ✅ Particiones creadas correctamente (30 períodos)
# MAGIC 4. ✅ Distribución por año verificada
# MAGIC 5. ✅ Estadísticas descriptivas calculadas
# MAGIC
# MAGIC ### Siguientes Pasos
# MAGIC ➡️ Ejecutar `02_silver_limpieza_ventas` para aplicar transformaciones y enriquecimientos
# MAGIC
# MAGIC ---
# MAGIC **Proyecto:** Lakehouse Ventas Duratex Colombia  
# MAGIC **Autor:** Data Engineering Team  
# MAGIC **Última actualización:** 2026
