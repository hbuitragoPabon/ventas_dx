# ventas_dx
# Análisis de Ventas Históricas Duratex Colombia 2024–2026
### Arquitectura Medallón en Databricks — Big Data | Especialización en Analítica de Datos | UNAULA I-2026

---

## Información del Proyecto

| Campo | Detalle |
|---|---|
| **Estudiante** | Gisella Martínez y Hector Buitrago |
| **Docente** | Yeis Taborda |
| **Institución** | Especialización en Analítica de Datos — Facultad de Ingeniería — UNAULA |
| **Período** | I - 2026 |
| **Fecha** | 26 de junio de 2026 |
| **Dataset** | `Detalle_ventas_2024_2026.xlsx` — 179,075 registros — 48 columnas |
| **Plataforma** | Databricks |
| **Catálogo** | `ventas_duratex` |

---

## 1. Caso de Negocio

### 1.1 Descripción del Problema

Duratex Colombia S.A.S. es fabricante y distribuidora de tableros de madera compuesta (MDP y MDF) con presencia en mercados nacionales e internacionales. Su sistema ERP genera transacciones comerciales detalladas por referencia, cliente, bodega y vendedor. Con 179,075 registros de ventas entre enero 2024 y junio 2026, la empresa enfrenta el desafío de aprovechar este activo de datos para mejorar decisiones estratégicas.

El análisis actual se realiza manualmente en hojas de cálculo, sin capacidad de detectar patrones de rentabilidad, clasificar automáticamente devoluciones ni proyectar tendencias. Una arquitectura Big Data en Databricks permitirá escalar este análisis y convertirlo en inteligencia de negocio continua.

### 1.2 Objetivos del Proyecto

**Objetivo General:** Desarrollar una solución Big Data sobre Databricks con Arquitectura Medallón (Bronze, Silver, Gold) para procesar el histórico de ventas Duratex 2024-2026, habilitando análisis descriptivos, visualizaciones y modelos predictivos que soporten la toma de decisiones comerciales.

**Objetivos Específicos:**

- Ingestar el dataset (179,075 registros, 48 columnas) a la capa Bronze sin modificaciones.
- Limpiar y enriquecer en Silver: clasificar documentos, calcular márgenes, estandarizar strings.
- Construir tablas Gold agregadas por período, vendedor, bodega, línea y mercado.
- Clasificar automáticamente devoluciones vs ventas reales.
- Generar visualizaciones clave de análisis comercial.
- Implementar clustering de clientes y forecast de ventas.
- Desplegar dashboard para el equipo comercial.

---

## 2. Relación Beneficio / Coste

### 2.1 Análisis Económico

| Concepto | Situación Actual | Con Solución Big Data |
|---|---|---|
| Tiempo de reporte mensual | 16 horas/mes (analista) | < 5 min (automatizado) |
| Analistas requeridos | 2 analistas full-time | 0.5 FTE supervisión |
| Costo horas-hombre (año) | ~ COP $48,000,000 | ~ COP $12,000,000 |
| Detección de devoluciones | Manual, reactiva | Automática, tiempo real |
| Análisis de rentabilidad | No disponible | Dashboard siempre activo |

### 2.2 Retorno de Inversión

| Concepto | Valor Estimado (COP) |
|---|---|
| Costo Databricks (académico) | $0 |
| Horas de desarrollo (80h × $50,000) | $4,000,000 |
| Ahorro anual en horas-hombre | $36,000,000 / año |
| Valor por mejora en decisiones (1% ventas) | ~ $17,309,694,489 / año |
| **ROI estimado primer año** | **> 1,200%** |

> Nota: Las ventas totales 2024-2026 suman COP $1.73 billones.

---

## 3. Arquitectura Propuesta

```
FUENTES DE DATOS → INGESTA → MEDALLÓN LAKEHOUSE → CONSUMO
ERP Duratex        Auto Loader   BRONZE → SILVER → GOLD   Databricks SQL
(Excel/JDBC)                     (Delta Lake)              Power BI
                                                           Genie AI / MLflow
```

### Componentes Técnicos

| Componente | Descripción |
|---|---|
| **Plataforma** | Databricks Lakehouse |
| **Almacenamiento** | Delta Lake (Parquet + transaction log): ACID, time travel, schema enforcement |
| **Procesamiento** | Apache Spark (PySpark): distribuido, escalable |
| **Orquestación** | Databricks Workflows: pipelines Bronze→Silver→Gold |
| **Gobernanza** | Unity Catalog: linaje, control de acceso, auditoría |
| **IA Conversacional** | Databricks Genie Code: Text-to-SQL sobre tablas Gold |
| **Visualización** | Databricks SQL Dashboards + Power BI |
| **ML/MLOps** | MLflow: tracking de experimentos, versionado, Serving Endpoints |


<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/5d8754ce-903f-441f-94b9-5b0bd1591f4e" />


---

## 4. Pipeline de Ingesta de Datos

### 4.1 Perfil del Dataset

| Dimensión | Valor | Detalle |
|---|---|---|
| Total registros | 179,075 | Transacciones de venta y devolución 2024-2026 |
| Total columnas | 48 | Variables por transacción: IDs, valores, descriptores |
| Períodos cubiertos | 30 meses | Enero 2024 → Junio 2026 (formato AAAAMM) |
| Compañías | 2 | Compañía 1 y Compañía 2 (Duratex Colombia) |
| Mercados | 2 | Interno (NAL) y Externo (EXT) |
| Tipos de documento | 4 | EFN, EFE (facturas) \| ENN, ENE (notas crédito) |
| Bodegas/CEDI | 17 | Centros de distribución |
| Vendedores activos | 19 | Con nombre completo e ID |
| Líneas de producto | MDP, MDF | Tableros partícula y fibra de densidad media |
| Sublíneas | 9 | Melamina, Desnudo, Duralam, Formaleta, otros |
| Calibres únicos | 13 | 3mm a 30mm |
| Registros negativos | 2,635 | 1.47% — devoluciones/notas crédito |
| Valores nulos | 0 | Dataset sin valores faltantes |
| Venta neta total | $1.73 billones COP | Incluye devoluciones |

### 4.2 Limpiezas y Transformaciones (Capa Silver)

| Hallazgo | Descripción | Solución |
|---|---|---|
| Strings con espacios | `TIPO_CLIENTE ("NAL ")`, `UNIDAD_MEDIDA ("UND ")`, `BODEGA ("CTT  ")` | `.trim()` en todas las columnas `StringType()` |
| Tipo documento sin descripción | EFN, EFE, ENN, ENE sin campo descriptivo | Crear `TIPO_TRANSACCION`: "VENTA" vs "DEVOLUCION" |
| 2,635 valores negativos | `VLR_VENTA_NETA < 0` sin clasificar | `IS_DEVOLUCION` flag booleano |
| LINEA = "NULL" (string) | Cadena literal "NULL" en vez de nulo real | `when(col("LINEA")=="NULL", None)` |
| Sin margen calculado | No existe columna de margen | `MARGEN_BRUTO = VLR_VENTA_NETA - (COSTO_PROMEDIO * CANT_FACTURADA)` |
| PERIODO en formato entero | 202401 dificulta agrupaciones | Crear `ANO_PERIODO`, `MES_PERIODO`, `TRIMESTRE`, `FECHA_PERIODO` |
| COSTO_PROMEDIO = 0 | Margen aparente del 100% | Flag `COSTO_INVALIDO = True` |

### 4.3 Análisis Posibles

| # | Análisis | Descripción |
|---|---|---|
| 1 | Tendencia mensual/anual | YoY 2024 vs 2025 vs 2026, estacionalidad, meses pico |
| 2 | Participación por línea | MDP vs MDF: % ventas, margen, evolución trimestral |
| 3 | Ranking vendedores | Top 5 por venta neta, margen, volumen y # facturas |
| 4 | Mercado interno vs externo | Comparativo NAL/EXT: volúmenes, precios, rentabilidad |
| 5 | Análisis de devoluciones | Tasa por vendedor, producto y período |
| 6 | Rentabilidad por referencia | Margen bruto por SKU — productos estrella vs pérdida |
| 7 | Análisis por bodega/CEDI | Volumen y margen por centro de distribución |
| 8 | Segmentación de clientes | K-Means RFM: Champion, Leal, En Riesgo, Inactivo |
| 9 | Forecast de ventas | Modelo mensual por línea — proyección 6 meses |
| 10 | Análisis de descuentos | `VLR_DSCTO_LIN / venta_bruta` = % descuento por vendedor |


<img width="1014" height="480" alt="image" src="https://github.com/user-attachments/assets/5bb41e18-886e-425e-b6e5-52a8e9053f22" />

<img width="990" height="673" alt="image" src="https://github.com/user-attachments/assets/0a07ff35-59cb-4529-96a1-ad9c5c5d0d7b" />

<img width="996" height="683" alt="image" src="https://github.com/user-attachments/assets/68495743-8a22-4558-8cdc-a7b21fb24539" />

<img width="1015" height="596" alt="image" src="https://github.com/user-attachments/assets/e9790ad0-2976-4b48-9ac5-17084a011581" />

<img width="972" height="757" alt="image" src="https://github.com/user-attachments/assets/62d310d8-e562-4c86-9cf6-20ecda4fced9" />

<img width="975" height="670" alt="image" src="https://github.com/user-attachments/assets/104c87a1-e081-4592-9b5e-8c3e7efe66ea" />

<img width="981" height="753" alt="image" src="https://github.com/user-attachments/assets/b0d9c5e7-bccd-4fa1-874d-d9bcc458b429" />

<img width="992" height="788" alt="image" src="https://github.com/user-attachments/assets/e4f6767e-768b-4d5a-9fe4-602201244721" />











### 4.4 Estrategia Medallón

#### 🥉 CAPA BRONZE — Datos Crudos

- Fuente: `/Volumes/ventas_duratex/bronze/landing_zone/Detalle_ventas_2024_2026.xlsx`
- Tabla: `ventas_duratex.bronze.ventas_raw`
- **CERO transformaciones** — datos exactamente como vienen del ERP
- Columnas de auditoría: `ingestion_timestamp`, `source_file`, `batch_id`
- Particionado por `ANO` y `PERIODO`

#### 🥈 CAPA SILVER — Limpieza y Enriquecimiento

- Fuente: `ventas_duratex.bronze.ventas_raw`
- Tabla: `ventas_duratex.silver.ventas_limpias`
- Todas las transformaciones identificadas en 4.2
- Particionado por `ANO_PERIODO`, `MES_PERIODO`

#### 🥇 CAPA GOLD — Datos de Negocio

| Tabla | Descripción |
|---|---|
| `ventas_duratex.gold.ventas_por_periodo` | Agregado por fecha y mercado |
| `ventas_duratex.gold.ventas_por_vendedor` | Ranking con margen y devoluciones |
| `ventas_duratex.gold.ventas_por_producto` | Top SKUs con rentabilidad |
| `ventas_duratex.gold.ventas_por_bodega` | Desempeño por CEDI con YoY |
| `ventas_duratex.gold.clientes_rfm` | Segmentación RFM de clientes |

---

## 5. Modelos de Ciencia de Datos

### 5.1 Análisis Descriptivo

- Estadísticas univariadas de `VLR_VENTA_NETA`, `MARGEN_BRUTO`, `CANT_FACTURADA`, `PCT_MARGEN`
- Distribución por mercado y línea de producto
- Serie temporal mensual con media móvil de 3 meses
- Boxplot de márgenes por sublínea
- Heatmap de estacionalidad (mes × año)
- Top 10 clientes por venta neta 2024-2026
- Análisis de devoluciones por vendedor y período
- Comparativo YoY por bodega y mercado

### 5.2 Modelos Propuestos

| Modelo | Tipo | Objetivo | Métrica |
|---|---|---|---|
| K-Means (RFM) | No supervisado | Segmentación de clientes | Silhouette Score |
| Prophet | Serie temporal | Forecast de ventas mensuales a 6 meses | MAPE, RMSE |
| Regresión Lineal | Supervisado | Predicción de margen bruto | R², MAE |
| Isolation Forest | Detección anomalías | Identificar transacciones atípicas | Precision/Recall |
| Apriori | Minería de datos | Patrones de co-compra de referencias | Support, Lift |

---

## 6. App / Visualización

### 6.1 Serving Endpoint

El modelo Prophet se desplegará como Serving Endpoint en Databricks Model Serving, registrado en MLflow. Recibe período y línea de producto, retorna proyección con intervalos de confianza en JSON.

### 6.2 Dashboard Databricks SQL

| Panel | Contenido |
|---|---|
| KPI Summary | Ventas YTD, Margen Promedio, # Facturas, Tasa Devolución |
| Tendencia Mensual | Línea de tiempo ventas + margen 2024-2026 con media móvil |
| Participación por Línea | Donut chart MDP vs MDF |
| Top 10 Vendedores | Barras horizontales con colormap de % margen |
| Heatmap Estacionalidad | Ventas por mes × año |
| Análisis Devoluciones | Tasa por vendedor y período |
| Mercado Int. vs Ext. | Comparativo volúmenes y márgenes |
| Forecast | Proyección 6 meses con banda de confianza 80% |

---

## Estructura del Repositorio

```
ventas_duratex/
├── README.md
├── notebooks/
│   ├── 01_bronze_ingesta_ventas.py
│   ├── 02_silver_limpieza_ventas.py
│   ├── 03_gold_tablas_negocio.py
│   ├── 04_analisis_visualizaciones.py
│   └── 05_modelos_ciencia_datos.py
├── docs/
│   └── Proyecto_BigData_Ventas_Duratex.pdf
└── .github/
    └── workflows/
```

---

*Especialización en Analítica de Datos | Facultad de Ingeniería | UNAULA | Período I-2026*
