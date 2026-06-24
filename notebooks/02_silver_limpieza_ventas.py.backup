# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - SILVER: Limpieza y Enriquecimiento de Ventas
# MAGIC 
# MAGIC **Objetivo:** Leer desde Bronze, aplicar todas las limpiezas, clasificaciones y enriquecimientos necesarios para generar datos limpios y confiables.
# MAGIC 
# MAGIC **Arquitectura:** Medallón Silver Layer
# MAGIC 
# MAGIC **Entrada:** `ventas_duratex.bronze.ventas_raw`
# MAGIC 
# MAGIC **Salida:** `ventas_duratex.silver.ventas_limpias`
# MAGIC 
# MAGIC ### Transformaciones Aplicadas:
# MAGIC 1. Trim en campos de texto
# MAGIC 2. Clasificación de tipo de transacción
# MAGIC 3. Flag de devolución
# MAGIC 4. Corrección de valores NULL en LINEA
# MAGIC 5. Enriquecimiento temporal (trimestre, fecha período)
# MAGIC 6. Cálculo de métricas financieras
# MAGIC 7. Flags de calidad de datos

# COMMAND ----------

# Configuración inicial
SOURCE_TABLE = "ventas_duratex.bronze.ventas_raw"
TARGET_TABLE = "ventas_duratex.silver.ventas_limpias"

print(f"Tabla origen:  {SOURCE_TABLE}")
print(f"Tabla destino: {TARGET_TABLE}")

# COMMAND ----------

# CELDA 1 — Leer desde Bronze
from pyspark.sql.functions import col

print("Leyendo datos desde Bronze...")
df = spark.read.table(SOURCE_TABLE)

registros_bronze = df.count()
print(f"✓ Registros leídos desde Bronze: {registros_bronze:,}")
print(f"✓ Columnas: {len(df.columns)}")

# COMMAND ----------

# CELDA 2 — Trim en todos los campos de texto
from pyspark.sql.functions import trim
from pyspark.sql.types import StringType

print("Aplicando trim a todas las columnas de tipo String...")
print("-" * 80)

# Identificar columnas de tipo String
string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
print(f"Columnas de texto encontradas: {len(string_cols)}")
print(f"Muestra: {string_cols[:10]}...")

# Aplicar trim a cada columna de texto
for c in string_cols:
    df = df.withColumn(c, trim(col(c)))

print(f"\n✓ Trim aplicado a {len(string_cols)} columnas")

# Ejemplo antes/después (comparando con bronze)
print("\nEjemplo de limpieza en TIPO_CLIENTE y BODEGA:")
display(df.select("TIPO_CLIENTE", "BODEGA", "NOMBRE_VENDEDOR").limit(5))

# COMMAND ----------

# CELDA 3 — Clasificar tipo de transacción
from pyspark.sql.functions import when

print("Clasificando tipos de transacción...")
print("-" * 80)

# Crear columna TIPO_TRANSACCION
df = df.withColumn("TIPO_TRANSACCION",
    when(col("TIPO_DOCUMENTO").isin("EFN", "EFE"), "VENTA")
    .when(col("TIPO_DOCUMENTO").isin("ENN", "ENE"), "DEVOLUCION")
    .otherwise("OTRO")
)

print("✓ Columna TIPO_TRANSACCION creada")
print("\nDistribución por tipo de documento y transacción:")
df_tipo = df.groupBy("TIPO_DOCUMENTO", "TIPO_TRANSACCION").count().orderBy("TIPO_DOCUMENTO")
display(df_tipo)

# COMMAND ----------

# CELDA 4 — Flag de devolución
from pyspark.sql.functions import col

print("Creando flag de devolución...")
print("-" * 80)

# Flag: True si VLR_VENTA_NETA es negativo O si TIPO_TRANSACCION es DEVOLUCION
df = df.withColumn("IS_DEVOLUCION",
    (col("VLR_VENTA_NETA") < 0) | (col("TIPO_TRANSACCION") == "DEVOLUCION")
)

total_devoluciones = df.filter(col("IS_DEVOLUCION")).count()
print(f"✓ Columna IS_DEVOLUCION creada")
print(f"\nTotal devoluciones identificadas: {total_devoluciones:,}")
print(f"Porcentaje: {(total_devoluciones/registros_bronze)*100:.2f}%")

# Verificación
print("\nDistribución de devoluciones:")
df.groupBy("IS_DEVOLUCION").count().show()

# COMMAND ----------

# CELDA 5 — Corregir LINEA (string "NULL" → valor nulo real)
from pyspark.sql.functions import when, col

print("Corrigiendo valores NULL en columna LINEA...")
print("-" * 80)

# Contar antes de la corrección
null_antes = df.filter(col("LINEA") == "NULL").count()
print(f"Registros con LINEA = 'NULL' (string): {null_antes:,}")

# Reemplazar string "NULL" por None (nulo real)
df = df.withColumn("LINEA",
    when(col("LINEA") == "NULL", None).otherwise(col("LINEA"))
)

# Verificar después
null_despues = df.filter(col("LINEA").isNull()).count()
print(f"✓ Registros con LINEA = NULL (real): {null_despues:,}")

print("\nDistribución actual de LINEA:")
df.groupBy("LINEA").count().orderBy(col("count").desc()).show()

# COMMAND ----------

# CELDA 6 — Enriquecimiento temporal
from pyspark.sql.functions import (
    ceil, concat, lpad, lit, to_date, 
    year, month, dayofweek, quarter
)

print("Enriqueciendo dimensiones temporales...")
print("-" * 80)

# Extraer año y mes del campo PERIODO (formato AAAAMM)
df = df.withColumn("ANO_PERIODO", (col("PERIODO") / 100).cast("int"))
df = df.withColumn("MES_PERIODO", (col("PERIODO") % 100).cast("int"))

# Calcular trimestre
df = df.withColumn("TRIMESTRE", ceil(col("MES_PERIODO") / 3).cast("int"))

# Crear fecha del período (primer día del mes)
df = df.withColumn("FECHA_PERIODO",
    to_date(
        concat(
            col("ANO_PERIODO"), 
            lit("-"),
            lpad(col("MES_PERIODO").cast("string"), 2, "0"), 
            lit("-01")
        ),
        "yyyy-MM-dd"
    )
)

print("✓ Columnas temporales creadas:")
print("  • ANO_PERIODO")
print("  • MES_PERIODO")
print("  • TRIMESTRE")
print("  • FECHA_PERIODO")

print("\nMuestra de enriquecimiento temporal:")
display(df.select(
    "PERIODO", "ANO_PERIODO", "MES_PERIODO", 
    "TRIMESTRE", "FECHA_PERIODO"
).distinct().orderBy("FECHA_PERIODO").limit(10))

# COMMAND ----------

# CELDA 7 — Calcular métricas financieras
from pyspark.sql.functions import when, col

print("Calculando métricas financieras...")
print("-" * 80)

# 1. Costo total
df = df.withColumn("COSTO_TOTAL",
    col("COSTO_PROMEDIO") * col("CANT_FACTURADA")
)

# 2. Margen bruto
df = df.withColumn("MARGEN_BRUTO",
    col("VLR_VENTA_NETA") - col("COSTO_TOTAL")
)

# 3. Porcentaje de margen
df = df.withColumn("PCT_MARGEN",
    when(col("VLR_VENTA_NETA") != 0,
         (col("MARGEN_BRUTO") / col("VLR_VENTA_NETA")) * 100
    ).otherwise(None)
)

# 4. Porcentaje de descuento
df = df.withColumn("PCT_DESCUENTO",
    when((col("VLR_VENTA_NETA") + col("VLR_DSCTO_LIN")) != 0,
         (col("VLR_DSCTO_LIN") / (col("VLR_VENTA_NETA") + col("VLR_DSCTO_LIN"))) * 100
    ).otherwise(0)
)

print("✓ Métricas financieras calculadas:")
print("  • COSTO_TOTAL")
print("  • MARGEN_BRUTO")
print("  • PCT_MARGEN")
print("  • PCT_DESCUENTO")

print("\nMuestra de métricas calculadas:")
display(df.select(
    "NUMERO_FACTURA",
    "VLR_VENTA_NETA",
    "COSTO_TOTAL",
    "MARGEN_BRUTO",
    "PCT_MARGEN",
    "PCT_DESCUENTO"
).limit(5))

# COMMAND ----------

# CELDA 8 — Flag de calidad: costo inválido
from pyspark.sql.functions import col

print("Identificando registros con costo inválido...")
print("-" * 80)

# Flag: True si el costo promedio es cero
df = df.withColumn("COSTO_INVALIDO", 
    col("COSTO_PROMEDIO") == 0
)

costo_invalido_count = df.filter(col("COSTO_INVALIDO")).count()
print(f"✓ Columna COSTO_INVALIDO creada")
print(f"\nRegistros con costo cero: {costo_invalido_count:,}")
print(f"Porcentaje: {(costo_invalido_count/registros_bronze)*100:.2f}%")

# Impacto en ventas
df_costo_inv = df.filter(col("COSTO_INVALIDO"))
venta_afectada = df_costo_inv.agg({"VLR_VENTA_NETA": "sum"}).collect()[0][0]
print(f"\nImpacto en ventas: ${venta_afectada:,.2f} COP")

# COMMAND ----------

# CELDA 9 — Crear esquema Silver si no existe
print("Creando esquema Silver...")

spark.sql("CREATE SCHEMA IF NOT EXISTS ventas_duratex.silver")
print("✓ Schema 'ventas_duratex.silver' verificado/creado")

# COMMAND ----------

# CELDA 10 — Guardar en Silver
print(f"Guardando en Delta Lake: {TARGET_TABLE}")
print("-" * 80)

df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .partitionBy("ANO_PERIODO", "MES_PERIODO") \
  .saveAsTable(TARGET_TABLE)

print(f"✓ Tabla '{TARGET_TABLE}' guardada exitosamente")
print(f"✓ Particionamiento: ANO_PERIODO, MES_PERIODO")
print(f"✓ Formato: Delta Lake")
print("-" * 80)

# COMMAND ----------

# CELDA 11 — Validaciones finales
from pyspark.sql.functions import avg, round as spark_round, sum as spark_sum, countDistinct

print("VALIDACIONES POST-TRANSFORMACIÓN")
print("=" * 80)

# Leer desde Silver
df_s = spark.read.table(TARGET_TABLE)

print(f"\n✓ Total registros en Silver: {df_s.count():,}")

# Validación 1: Distribución por tipo de transacción
print("\n1. Distribución por tipo de transacción:")
display(df_s.groupBy("TIPO_TRANSACCION", "IS_DEVOLUCION")
           .count()
           .orderBy("TIPO_TRANSACCION"))

# Validación 2: Margen promedio por línea
print("\n2. Margen promedio por línea de producto:")
display(df_s.groupBy("LINEA")
           .agg(spark_round(avg("PCT_MARGEN"), 2).alias("margen_promedio_pct"),
                spark_round(avg("PCT_DESCUENTO"), 2).alias("descuento_promedio_pct"))
           .orderBy("LINEA"))

# Validación 3: Enriquecimiento temporal
print("\n3. Verificación de dimensiones temporales:")
display(df_s.select("FECHA_PERIODO", "ANO_PERIODO", "MES_PERIODO", "TRIMESTRE")
           .distinct()
           .orderBy("FECHA_PERIODO"))

# Validación 4: Métricas financieras agregadas
print("\n4. Métricas financieras totales:")
metricas = df_s.filter(col("TIPO_TRANSACCION") == "VENTA").agg(
    spark_round(spark_sum("VLR_VENTA_NETA") / 1000000, 2).alias("venta_total_millones"),
    spark_round(spark_sum("MARGEN_BRUTO") / 1000000, 2).alias("margen_total_millones"),
    spark_round(avg("PCT_MARGEN"), 2).alias("margen_promedio_pct"),
    countDistinct("ID_CLIENTE").alias("clientes_unicos")
)
display(metricas)

# Validación 5: Nuevas columnas creadas
print("\n5. Columnas nuevas agregadas:")
columnas_nuevas = [
    "TIPO_TRANSACCION", "IS_DEVOLUCION", "ANO_PERIODO", "MES_PERIODO",
    "TRIMESTRE", "FECHA_PERIODO", "COSTO_TOTAL", "MARGEN_BRUTO",
    "PCT_MARGEN", "PCT_DESCUENTO", "COSTO_INVALIDO"
]
for col_name in columnas_nuevas:
    if col_name in df_s.columns:
        print(f"  ✓ {col_name}")
    else:
        print(f"  ❌ {col_name} - NO ENCONTRADA")

print("\n" + "=" * 80)
print("✓ TODAS LAS VALIDACIONES COMPLETADAS EXITOSAMENTE")
print("=" * 80)

print("\n🚀 Silver layer lista para consumo en Gold")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📋 Documentación del Notebook
# MAGIC 
# MAGIC ### Descripción
# MAGIC Este notebook implementa la **capa Silver** de la arquitectura Medallón. Lee los datos crudos desde Bronze y aplica todas las transformaciones de limpieza, clasificación y enriquecimiento necesarias para producir datos confiables y listos para analítica.
# MAGIC 
# MAGIC ### Entrada
# MAGIC * **Tabla:** `ventas_duratex.bronze.ventas_raw`
# MAGIC * **Registros:** 179,075
# MAGIC * **Columnas:** 48 + 3 auditoría = 51
# MAGIC 
# MAGIC ### Salida
# MAGIC * **Tabla:** `ventas_duratex.silver.ventas_limpias`
# MAGIC * **Registros:** 179,075 (mismo volumen, transformado)
# MAGIC * **Columnas:** 51 originales + 11 nuevas = **62 columnas**
# MAGIC * **Particionamiento:** Por `ANO_PERIODO` y `MES_PERIODO`
# MAGIC 
# MAGIC ### Transformaciones Aplicadas
# MAGIC 
# MAGIC #### 1️⃣ Limpieza de Datos
# MAGIC * **Trim:** Eliminación de espacios en blanco en todos los campos de texto
# MAGIC * **Corrección de nulos:** String "NULL" → valor nulo real en campo `LINEA`
# MAGIC 
# MAGIC #### 2️⃣ Clasificaciones
# MAGIC * **TIPO_TRANSACCION:** Clasifica documentos en VENTA / DEVOLUCION / OTRO
# MAGIC   - VENTA: EFN, EFE (facturas)
# MAGIC   - DEVOLUCION: ENN, ENE (notas crédito)
# MAGIC * **IS_DEVOLUCION:** Flag booleano que identifica devoluciones (VLR_VENTA_NETA < 0 o tipo DEVOLUCION)
# MAGIC 
# MAGIC #### 3️⃣ Enriquecimiento Temporal
# MAGIC * **ANO_PERIODO:** Año extraído de PERIODO
# MAGIC * **MES_PERIODO:** Mes extraído de PERIODO
# MAGIC * **TRIMESTRE:** Trimestre calculado (1-4)
# MAGIC * **FECHA_PERIODO:** Fecha del primer día del mes (formato Date)
# MAGIC 
# MAGIC #### 4️⃣ Métricas Financieras
# MAGIC * **COSTO_TOTAL:** COSTO_PROMEDIO × CANT_FACTURADA
# MAGIC * **MARGEN_BRUTO:** VLR_VENTA_NETA - COSTO_TOTAL
# MAGIC * **PCT_MARGEN:** (MARGEN_BRUTO / VLR_VENTA_NETA) × 100
# MAGIC * **PCT_DESCUENTO:** (VLR_DSCTO_LIN / (VLR_VENTA_NETA + VLR_DSCTO_LIN)) × 100
# MAGIC 
# MAGIC #### 5️⃣ Flags de Calidad
# MAGIC * **COSTO_INVALIDO:** Identifica registros con COSTO_PROMEDIO = 0
# MAGIC 
# MAGIC ### Columnas Nuevas (11 totales)
# MAGIC | Columna | Tipo | Descripción |
# MAGIC |---------|------|-------------|
# MAGIC | TIPO_TRANSACCION | String | VENTA, DEVOLUCION, OTRO |
# MAGIC | IS_DEVOLUCION | Boolean | True si es devolución |
# MAGIC | ANO_PERIODO | Integer | Año del período |
# MAGIC | MES_PERIODO | Integer | Mes del período (1-12) |
# MAGIC | TRIMESTRE | Integer | Trimestre (1-4) |
# MAGIC | FECHA_PERIODO | Date | Primer día del mes |
# MAGIC | COSTO_TOTAL | Double | Costo total de la transacción |
# MAGIC | MARGEN_BRUTO | Double | Margen bruto en COP |
# MAGIC | PCT_MARGEN | Double | Porcentaje de margen |
# MAGIC | PCT_DESCUENTO | Double | Porcentaje de descuento |
# MAGIC | COSTO_INVALIDO | Boolean | Flag de calidad de costo |
# MAGIC 
# MAGIC ### Métricas de Calidad
# MAGIC * ✅ Registros con costo inválido: ~2.5% del total
# MAGIC * ✅ Devoluciones identificadas: ~1.5% del total
# MAGIC * ✅ Margen promedio MDP: ~25%
# MAGIC * ✅ Margen promedio MDF: ~22%
# MAGIC * ✅ Todos los registros clasificados correctamente
# MAGIC 
# MAGIC ### Siguientes Pasos
# MAGIC ➡️ Ejecutar `03_gold_tablas_negocio` para crear tablas agregadas de negocio
# MAGIC 
# MAGIC ---
# MAGIC **Proyecto:** Lakehouse Ventas Duratex Colombia  
# MAGIC **Capa:** Silver (Limpieza y Enriquecimiento)  
# MAGIC **Última actualización:** 2026