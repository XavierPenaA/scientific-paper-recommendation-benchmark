# Documento de Diseño de Investigación y Software (RSDD)

**Título del paper:**
*"A Reproducible Multi-Criteria Analysis of Recommendation Paradigms for Scientific Literature"*

Este documento es la fuente de verdad del proyecto. Define la pregunta de investigación, la arquitectura técnica, los experimentos, las métricas, el plan de análisis y el cronograma de ejecución. Todo lo que existe en el repositorio —scripts, configs, resultados, paper— tiene su origen aquí.

---

## 1. Pregunta de Investigación y Objetivos

### Pregunta central

> ¿Cómo se comportan los distintos paradigmas de recomendación en el dominio de literatura científica, y cuándo conviene cada uno según las características del dataset, los recursos computacionales disponibles y los objetivos del sistema?

Esta pregunta distingue el trabajo de un benchmark simple. No respondemos únicamente "¿qué modelo tiene mayor Recall?", sino que analizamos el **comportamiento** de cada paradigma bajo distintas condiciones, estableciendo guías prácticas basadas en evidencia estadística.

### Preguntas de Investigación (RQ)

**RQ1 — Precisión vs. Complejidad:**
¿Superan estadísticamente los modelos profundos (Transformers, GNNs) a los baselines clásicos en la tarea de recomendar artículos científicos?

- *Hipótesis:* Los modelos GNN (LightGCN) superarán a los clásicos en precisión en datasets densos, pero la ventaja se reducirá significativamente en escenarios dispersos (sparse).

**RQ2 — Consistencia entre plataformas:**
¿Se mantienen consistentes las conclusiones sobre el comportamiento de cada paradigma en múltiples plataformas independientes de gestión de literatura científica (CiteULike y BibSonomy)?

- *Hipótesis:* Las diferencias de densidad y comportamiento de usuario entre plataformas afectarán el ranking relativo de los modelos, especialmente en modelos sensibles a la sparsity como LightGCN. Sin embargo, las tendencias generales (qué familias son más robustas, más diversas, más eficientes) se mantendrán consistentes.

**RQ3 — Robustez a datos escasos:**
¿Qué paradigma se degrada menos cuando solo se dispone de un porcentaje reducido de las interacciones de entrenamiento?

- *Hipótesis:* ItemKNN mostrará mayor robustez a la reducción de datos por no tener parámetros que sobreajustar; los modelos secuenciales serán los más sensibles.

**RQ4 — Trade-offs multicriterio:**
¿Existe un modelo que domine en todas las dimensiones (precisión, diversidad, robustez, costo computacional) o los trade-offs son inevitables?

- *Hipótesis:* No existe un paradigma universalmente óptimo; cada uno domina en un punto distinto de la Frontera de Pareto (precisión vs. costo) y la elección depende del contexto de uso.

---

## 2. Datasets

Se utilizan tres datasets del dominio de recomendación académica, permitiendo analizar la generalización de las conclusiones.

| Dataset | Usuarios | Items | Interacciones | Densidad | Fuente |
|---|---|---|---|---|---|
| CiteULike-a | 5,536 | 15,429 | 200,180 | 0.2344% | CiteULike (split A) |
| CiteULike-t | 3,527 | 6,339 | 77,546 | 0.3468% | CiteULike (split T) |
| BibSonomy | ~2,000–5,000 | ~30,000+ | TBD tras preprocesamiento | TBD | BibSonomy (social bookmarking académico) |

**Por qué tres datasets:** Dos datasets son insuficientes para afirmaciones de generalización — especialmente cuando ambos provienen de la misma plataforma (CiteULike). BibSonomy es una plataforma de marcadores sociales académicos completamente independiente, con comportamiento de usuario análogo (guardar papers de interés) pero origen distinto. Su inclusión permite afirmar que las conclusiones se mantienen consistentes en más de una fuente de datos académica.

**Por qué BibSonomy y no DBLP o Semantic Scholar:** BibSonomy es pequeño, completamente público (snapshots regulares disponibles en su sitio oficial), en formato de interacciones usuario-item directamente usable, y ya ha sido utilizado junto a CiteULike en papers previos de recomendación académica — lo que proporciona precedente bibliográfico. DBLP y Semantic Scholar son datasets de citación de millones de registros que requerirían semanas de preprocesamiento.

### Protocolo de Preprocesamiento

**Filtrado 5-core:** Se eliminan iterativamente usuarios con menos de 5 interacciones y papers con menos de 5 lectores, hasta que el dataset sea estable. Esto elimina ruido extremo sin sesgar el análisis.

**Formato de salida (RecBole `.inter`):**
```
user_id:token    item_id:token    timestamp:float
```

Las interacciones representan feedback implícito: el usuario agregó el paper a su biblioteca personal. No hay calificaciones explícitas.

### Protocolos de Evaluación

- **LOO (Leave-One-Out):** El último ítem de cada usuario va a test, el penúltimo a validación. Estándar histórico en papers de recomendación secuencial.
- **RS (Random Split 80/10/10):** Split aleatorio estratificado. Estándar más robusto estadísticamente.

Ambos protocolos se aplican a los tres datasets, generando 6 configuraciones de evaluación independientes.

---

## 3. Modelos — Seis Paradigmas Representativos

Se evalúan seis modelos que representan familias arquitectónicas distintas. El término **"representativos"** es intencional: no se pretende cubrir todos los paradigmas existentes (quedan fuera modelos de difusión y LLM-based recommenders — mencionados como trabajo futuro). MultiVAE se incluye por su prominencia como enfoque generativo ampliamente adoptado y reproducido en la literatura de RS.

| Paradigma | Modelo | Referencia |
|---|---|---|
| Filtrado colaborativo clásico | **ItemKNN** | Sarwar et al., WWW 2001 |
| Neural CF | **NeuMF** | He et al., WWW 2017 |
| Autoencoder Variacional | **MultiVAE** | Liang et al., WWW 2018 |
| Transformer unidireccional | **SASRec** | Kang & McAuley, ICDM 2018 |
| Transformer bidireccional | **BERT4Rec** | Sun et al., CIKM 2019 |
| Graph Neural Network | **LightGCN** | He et al., SIGIR 2020 |

**Justificación de MultiVAE:** Liang et al. (2018) demostraron que los modelos generativos basados en inferencia variacional superan a NeuMF en escenarios de feedback implícito y han sido extensamente adoptados como baseline en la literatura. Su inclusión cubre el paradigma generativo/probabilístico, complementando las familias de memoria (KNN), factorización (NeuMF), secuencial-Transformer (SASRec, BERT4Rec) y grafos (LightGCN).

### Hiperparámetros

Todos los modelos usan configuraciones derivadas de sus publicaciones originales o de la implementación oficial de RecBole para evitar la crítica de "baselines débiles". Ver archivos `configs/` para los valores exactos.

**Configuración base global (`configs/base.yaml`):**
- Semilla aleatoria: `2024` (fija para todos los experimentos)
- Métrica de validación: `NDCG@10`
- Early stopping: 10 épocas sin mejora
- Top-K: 5, 10, 20

---

## 4. Métricas de Evaluación — Tres Dimensiones

La evaluación cubre tres dimensiones complementarias. Esto es lo que transforma un benchmark en un análisis de comportamiento.

### Dimensión 1: Precisión de Ranking

Miden qué tan bien el modelo coloca los papers relevantes en el Top-K de la lista.

| Métrica | Descripción |
|---|---|
| Recall@K | Fracción de papers relevantes recuperados en el Top-K |
| NDCG@K | Calidad del ordenamiento — penaliza si el relevante aparece tarde |
| MAP@K | Precisión promedio para todos los usuarios |
| MRR@K | Recíproco del rank del primer ítem relevante |
| Precision@K | Fracción de recomendaciones en Top-K que son relevantes |

Con K = 5, 10, 20 para cada métrica.

### Dimensión 2: Calidad de las Recomendaciones

Miden si las recomendaciones son útiles más allá de la precisión pura.

| Métrica | Descripción |
|---|---|
| Coverage | Fracción del catálogo total que el modelo puede recomendar |
| Novelty | Tendencia a recomendar ítems menos conocidos (inverso de popularidad) |
| Diversity | Variedad intra-lista: qué tan distintos entre sí son los papers recomendados |

**Análisis de sesgo de popularidad:** Adicionalmente, para cada modelo se calcula la distribución de popularidad de sus recomendaciones — clasificando ítems como "populares" (top 20% más interaccionados en entrenamiento) vs. "de nicho" (bottom 50%). Esto permite responder: ¿el modelo tiende a recomendar siempre los papers más leídos, o descubre trabajo relevante menos conocido?

> **Nota sobre SHAP:** El análisis de explicabilidad basado en SHAP sobre embeddings latentes no se incluye en este trabajo. SHAP devuelve importancias sobre dimensiones abstractas del embedding (e.g., "dimensión 23 contribuyó +0.12") que no tienen significado semántico interpretable. El análisis de popularidad y diversity definidos arriba provee insights más directamente útiles para usuarios académicos reales.

### Dimensión 3: Costo Computacional

Mide la eficiencia real de cada paradigma.

| Métrica | Descripción |
|---|---|
| Parámetros entrenables | Número total de parámetros del modelo |
| Tiempo de entrenamiento | Segundos por época de entrenamiento |
| GPU RAM peak | Memoria GPU máxima utilizada (GB) |
| Inference throughput | Usuarios procesados por segundo (users/s) — más intuitivo que ms/user para comparar escalabilidad |

Estos datos se extraen directamente de los logs de RecBole y del monitoreo de GPU durante las corridas.

---

## 5. Análisis Estadístico

La comparación entre modelos se valida estadísticamente. Sin tests estadísticos, afirmaciones como "LightGCN obtuvo 0.418 vs 0.412 de SASRec" no son científicamente sostenibles.

### Tests a aplicar

1. **Friedman Test** (no-paramétrico, equivalente a ANOVA para rankings): Verifica si existe diferencia global estadísticamente significativa entre los 5 modelos.
2. **Kendall's W (Tamaño del Efecto Global)**: Medida de concordancia que indica la fuerza del efecto del test de Friedman (rango 0 a 1). Un $W$ cercano a 0 indica nula concordancia en los rankings de los algoritmos; un $W$ cercano a 1 indica que los modelos difieren de forma drástica y consistente. Se calcula como:
   $$W = \frac{\chi^2}{N(k - 1)}$$
   donde $\chi^2$ es el estadístico de Friedman, $N$ es el número de condiciones de prueba (observaciones) y $k$ es el número de modelos (5).
3. **Nemenyi Post-hoc Test**: Comparación por pares para identificar qué modelos son significativamente distintos entre sí (p < 0.05).
4. **Critical Difference (CD) Diagram**: Visualización estándar en papers de comparación de algoritmos (Demšar, 2006). Muestra visualmente qué modelos son estadísticamente equivalentes (conectados por una barra horizontal) y cuáles no.

### Implementación
```python
# scipy + scikit-posthocs
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Kendall's W calculation helper
def kendalls_w(chisq, n, k):
    return chisq / (n * (k - 1))

# Organizar resultados por modelo × condición (dataset × protocolo × K)
# Aplicar Friedman -> calcular Kendall's W -> si p < 0.05, aplicar Nemenyi
# Generar CD Diagram con sp.critical_difference_diagram()
```

---

## 6. Análisis de Robustez

**Objetivo:** Responder RQ3 — ¿qué paradigma se degrada menos con datos escasos?

**Procedimiento:** Para cada dataset, se generan subsets de entrenamiento con el 50% y el 20% de las interacciones (muestreo aleatorio estratificado por usuario, semilla fija). El test set permanece intacto.

```
Runs totales: 5 modelos × 3 datasets × 2 niveles (50%, 20%) = 30 corridas
(además de las 30 corridas base al 100%)
```

**Resultado esperado:** Curvas de degradación de NDCG@10 vs. porcentaje de datos disponibles, una por paradigma. Permite identificar el "punto de quiebre" de cada familia.

---

## 7. Figuras del Paper (9 figuras planificadas)

Las figuras son lo que vende el paper. Se planifican 9 figuras con propósito específico cada una.

| # | Figura | Descripción | Herramienta |
|---|---|---|---|
| Fig. 1 | **Pipeline completo** | Diagrama de flujo: Datasets → Preprocesamiento → 5 modelos → Entrenamiento → Evaluación → Análisis estadístico → Conclusiones | TikZ en LaTeX o matplotlib |
| Fig. 2 | **Heatmap de métricas** | Modelos (filas) vs. métricas (columnas), coloreado por valor. Más legible que una tabla plana. | seaborn |
| Fig. 3 | **Frontera de Pareto** | NDCG@10 (eje Y) vs. tiempo de entrenamiento en escala log (eje X). Un punto por modelo. Frontera de Pareto destacada. | matplotlib |
| Fig. 4 | **Curvas de robustez** | NDCG@10 (eje Y) vs. % de datos de entrenamiento (20%, 50%, 100%) en eje X. Una línea por paradigma. Probablemente la figura más impactante del paper. | matplotlib |
| Fig. 5 | **CD Diagram** | Diagrama de Diferencia Crítica generado de los tests de Nemenyi. Obligatorio en papers de comparación de algoritmos. | scikit-posthocs |
| Fig. 6 | **Popularity Bias** | Bar chart apilado: qué porcentaje de recomendaciones de cada modelo proviene del Top 20% (populares), Middle 30%, Long Tail 50%. Conecta con Novelty. | matplotlib |
| Fig. 7 | **Diversity Boxplot** | Boxplot de diversity score por modelo. Mucho más informativo que una tabla de medias. | matplotlib/seaborn |
| Fig. 8 | **Radar/Spider Chart** | Perfil multicriterio de cada paradigma en 5 dimensiones: precisión, costo, robustez, diversidad, novedad. | matplotlib |
| Fig. 9 | **Dataset Statistics** | Tabla visual con: Dataset, Users, Items, Interactions, Density, Avg interactions/user, **Avg interactions/item**. Diseño limpio como figura, no como tabla LaTeX plana. | matplotlib |

---

## 8. Tabla de Hiperparámetros (en lugar de ablation study)

> [!IMPORTANT]
> No se realiza ablation study. Razón: el paper estudia paradigmas, no propone un modelo nuevo. Hacer ablation solo de LightGCN (por ejemplo) generaria la pregunta: ¿por qué no de SASRec también? En cambio, se documenta explícitamente que todos los hiperparámetros siguen las recomendaciones de los papers originales o la implementación oficial de RecBole.

Tabla a incluir en el paper (Sección de Metodología):

| Modelo | Paper original | Hiperparámetros clave |
|---|---|---|
| ItemKNN | Sarwar et al. (2001) | k=100, similitud coseno, shrinkage=0 |
| NeuMF | He et al. (2017) | embeddings=64, capas MLP=[256,128,64], dropout=0.1, Adam lr=0.001 |
| SASRec | Kang & McAuley (2018) | capas=2, cabezas=2, hidden=64, max\_seq\_len=50 |
| BERT4Rec | Sun et al. (2019) | misma arquitectura que SASRec, mask\_ratio=0.2 |
| LightGCN | He et al. (2020) | embeddings=64, capas GCN=3, regularización L2=1e-4 |

Frase estándar para el paper:
> *"All hyperparameters follow the recommendations of the original publications and the official RecBole implementations to avoid introducing optimization bias toward any particular paradigm."*

---

## 9. Tabla de Decisión (Figura/Tabla de Cierre del Paper)

Esta tabla resume TODO el paper en una sola vista. Va al final de la sección de Discusión, justo antes de las Conclusiones.

| Escenario de uso | Paradigma recomendado | Justificación |
|---|---|---|
| Pocos datos de interacción | **ItemKNN** | Mayor robustez a data escasa; sin parámetros que sobreajustar |
| Máxima precisión, GPU disponible | **LightGCN** | Mayor NDCG@10 en datasets suficientemente densos |
| Solo CPU disponible | **ItemKNN** | Único paradigma sin requerimiento de GPU para escalar |
| Alta diversidad requerida | **BERT4Rec** | Mayor diversidad intra-lista según métricas de diversity |
| Descubrimiento de papers de nicho | **BERT4Rec** o **SASRec** | Mayor novelty; menor sesgo hacia papers populares |
| Sistema nuevo / cold-start | **ItemKNN** | Más robusto con 20% de interacciones disponibles |
| Balance general precisión/costo | **SASRec** o **NeuMF** | Compromiso razonable en todas las dimensiones |

*(Valores definitivos a completar con resultados reales)*

---

## 10. Tabla de Navegación por RQs (para la Introducción del paper)

Esta tabla va en la Introducción — le dice al lector desde el principio cómo está organizado el paper y qué responde cada parte.

| RQ | Experimento | Sección | Conclusión esperada |
|---|---|---|---|
| **RQ1** Precisión vs. Complejidad | Comparación de métricas de ranking en 3 datasets | Sección 4.1 | No todos los modelos profundos difieren significativamente de los clásicos |
| **RQ2** Consistencia entre plataformas | Resultados en CiteULike-a, CiteULike-t y BibSonomy | Sección 4.1 + 5.1 | Las tendencias generales se mantienen; la densidad del dataset modula la ventaja de GNNs |
| **RQ3** Robustez a datos escasos | Entrenamiento con 20%, 50%, 100% de interacciones | Sección 4.5 | ItemKNN es el paradigma más robusto; los modelos secuenciales son los más sensibles |
| **RQ4** Trade-offs multicriterio | Análisis de Pareto + tabla de trade-offs | Sección 5.2 | No existe un paradigma universalmente óptimo; la elección depende del contexto |

---

## 11. Visualizaciones de Análisis

### 7.1 Tabla de Trade-offs (Resumen Ejecutivo)

Tabla con ratings ★ (1-5) por paradigma en cada dimensión. Resume el paper completo en una sola figura.

| Paradigma | Precisión | Costo | Robustez | Diversidad | Novedad |
|---|---|---|---|---|---|
| ItemKNN | ★★★★ | ★★★★★ | ★★★★★ | ★★ | ★★ |
| NeuMF | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| SASRec | ★★★★ | ★★ | ★★ | ★★★★ | ★★★★ |
| BERT4Rec | ★★★★ | ★ | ★★ | ★★★★★ | ★★★★★ |
| LightGCN | ★★★★★ | ★ | ★★★ | ★★★ | ★★★ |

*(Valores preliminares — se actualizan con resultados reales)*

### 7.2 Frontera de Pareto (Precisión vs. Costo)

Gráfico de dispersión: eje X = tiempo de entrenamiento (log scale), eje Y = NDCG@10. Cada punto es un modelo. Se dibuja la frontera de Pareto, mostrando que ningún modelo domina en ambas dimensiones simultáneamente.

**Mensaje clave:** No existe un mejor modelo universal. ItemKNN domina en eficiencia. LightGCN domina en precisión. La elección depende del contexto.

### 7.3 Radar Chart Multicriterio

Perfil visual de cada paradigma en 5 dimensiones: precisión, costo, robustez, diversidad, novedad. Permite comparar "personalidades" de cada modelo de un vistazo.

### 7.4 Curvas de Degradación por Robustez

Líneas de NDCG@10 vs. % de datos de entrenamiento (20%, 50%, 100%) para cada paradigma. Responde visualmente a RQ3.

### 7.5 Critical Difference Diagram

Generado automáticamente desde los tests de Nemenyi. Muestra qué diferencias son estadísticamente significativas.

---

## 8. Demo — Aplicación de Demostración

**Naturaleza:** No es una contribución científica. Es una demostración de aplicabilidad de los hallazgos del paper.

**Funcionamiento:**
1. El usuario especifica su contexto: tamaño del dataset, recursos disponibles (GPU/CPU), objetivo prioritario (precisión / diversidad / velocidad)
2. La aplicación recomienda el paradigma más adecuado basándose en los hallazgos del paper
3. Muestra recomendaciones de ejemplo sobre CiteULike usando el modelo seleccionado

**Ubicación:** `app/` dentro del repositorio.

---

## 9. Estructura del Repositorio

```
paper-recommendation-benchmark/
│
├── datasets/
│   ├── citeulike-a/          # .inter, stats.csv
│   ├── citeulike-t/          # .inter, stats.csv
│   └── bibsonomy/            # .inter, stats.csv
│
├── configs/
│   ├── base.yaml             # Configuración global compartida
│   ├── ItemKNN.yaml
│   ├── NeuMF.yaml
│   ├── SASRec.yaml
│   ├── BERT4Rec.yaml
│   └── LightGCN.yaml
│
├── scripts/
│   ├── download_datasets.py  # Descarga y preprocesamiento
│   ├── run_benchmark.py      # Runner principal (train + eval + compile)
│   └── run_robustness.py     # Corridas con datos reducidos
│
├── analysis/
│   ├── statistical_tests.py  # Friedman + Nemenyi + CD Diagram
│   ├── popularity_bias.py    # Análisis de sesgo de popularidad
│   ├── pareto_frontier.py    # Frontera de Pareto (precisión vs. costo)
│   ├── radar_chart.py        # Radar multicriterio
│   └── robustness_curves.py  # Curvas de degradación
│
├── results/
│   ├── *.json                # Resultados individuales por corrida
│   ├── comparison_table.csv  # Tabla compilada de todos los resultados
│   └── robustness/           # Resultados de corridas con datos reducidos
│
├── app/
│   └── ...                   # Demo de selección de paradigma
│
└── paper/
    ├── main.tex
    ├── references.bib
    └── figures/              # Todos los gráficos generados
```

---

## 10. Cronograma de Ejecución

### FASE 1 — Completar entrenamientos base (EN PROCESO)

```powershell
# Ya completados:
# ItemKNN, NeuMF, SASRec, BERT4Rec en citeulike-a (LOO + RS)
# ItemKNN, NeuMF, SASRec en citeulike-t (LOO + RS)
# BERT4Rec en citeulike-t LOO

# Corriendo ahora:
.venv\Scripts\python.exe scripts/run_benchmark.py --model LightGCN --dataset citeulike-a --split loo

# Pendientes (secuenciales):
.venv\Scripts\python.exe scripts/run_benchmark.py --model LightGCN --dataset citeulike-a --split rs
.venv\Scripts\python.exe scripts/run_benchmark.py --model LightGCN --dataset citeulike-t --split loo
.venv\Scripts\python.exe scripts/run_benchmark.py --model LightGCN --dataset citeulike-t --split rs
.venv\Scripts\python.exe scripts/run_benchmark.py --model BERT4Rec --dataset citeulike-t --split rs
```

### FASE 2 — Tercer Dataset: BibSonomy

**Pasos de preprocesamiento:**
1. Descargar el snapshot público desde el sitio de BibSonomy (datos.bibsonomy.org o página de investigación de la Universidad de Kassel)
2. Filtrar solo interacciones de tipo `publication` (papers), descartando bookmarks web
3. Aplicar filtrado 5-core iterativo (mismo protocolo que CiteULike)
4. Exportar a formato RecBole `.inter` con columnas `user_id:token`, `item_id:token`, `timestamp:float`
5. Calcular y guardar estadísticas en `stats.csv`

```powershell
# Preprocesar BibSonomy
python scripts/download_datasets.py --dataset bibsonomy

# Correr todos los modelos en BibSonomy
python scripts/run_benchmark.py --all --dataset bibsonomy --split loo
python scripts/run_benchmark.py --all --dataset bibsonomy --split rs
```

### FASE 3 — Activar métricas extendidas y recompilar

Modificar `configs/base.yaml` para incluir `coverage`, `novelty`, `diversity` en la lista de métricas. Luego:

```powershell
.venv\Scripts\python.exe scripts/run_benchmark.py --compile
```

### FASE 4 — Extraer costos computacionales

```powershell
python analysis/extract_compute_costs.py
# Lee los logs de todas las corridas y genera results/compute_costs.csv
```

### FASE 5 — Análisis de sesgo de popularidad

```powershell
python analysis/popularity_bias.py
# Para cada modelo, calcula distribución de popularidad de recomendaciones
# Genera figures/popularity_bias.png
```

### FASE 6 — Corridas de robustez

```powershell
python scripts/run_robustness.py --levels 0.5 0.2 --datasets citeulike-a citeulike-t bibsonomy
# Corre los 5 modelos con 50% y 20% de datos de entrenamiento
# 30 corridas totales
```

### FASE 7 — Tests estadísticos

```powershell
python analysis/statistical_tests.py
# Friedman Test global
# Nemenyi post-hoc por pares
# Genera figures/cd_diagram.png
```

### FASE 8 — Visualizaciones finales

```powershell
python analysis/pareto_frontier.py   # figures/pareto.png
python analysis/radar_chart.py       # figures/radar.png
python analysis/robustness_curves.py # figures/robustness.png
```

### FASE 9 — Redacción del paper

Actualizar `paper/main.tex`:

1. Abstract y título
2. Introducción con nuevas RQs
3. Sección 2: Related Work
4. Sección 3: Metodología (datasets, modelos, métricas, tests estadísticos)
5. Sección 4: Resultados (5 subsecciones: precisión, costo, calidad, sesgo, robustez)
6. Sección 5: Discusión (Pareto, trade-offs, guía práctica por escenario)
7. Sección 6: Conclusiones y trabajo futuro

### FASE 10 — Demo App

Construir aplicación mínima en `app/` que implemente la lógica de selección de paradigma basada en contexto del usuario.

### FASE 11 — Repositorio final y sumisión

- Limpiar todos los scripts, añadir docstrings y README completo con instrucciones de reproducibilidad
- Tagear versión estable en GitHub
- Enviar paper a revista objetivo

---

## 11. Criterios de Validación

Un paso se considera finalizado cuando:

1. **Sin pérdida de datos:** Los datasets en `.inter` tienen el número exacto de filas y coinciden con `stats.csv`.
2. **Reproducibilidad:** Semilla `2024` fija en todos los experimentos. Cualquier corrida con los mismos parámetros produce los mismos resultados.
3. **Logs completos:** Cada corrida registra tiempo por época, pérdida de entrenamiento, y métricas de validación.
4. **Compilación limpia:** `comparison_table.csv` contiene el número correcto de filas (modelos × datasets × splits).
5. **Sin hardcoding:** Todos los paths son relativos al directorio `research/`. Sin rutas absolutas en ningún script.

---

## 12. Estrategia de Publicación

| Orden | Journal | Razón |
|---|---|---|
| 1 | **Expert Systems with Applications** (Elsevier, Q1, IF ~8.5) | Scope amplio que acepta estudios aplicados de IA/ML; revisores constructivos |
| 2 | **Information Sciences** o **Knowledge-Based Systems** | Si ESWA rechaza con comentarios útiles y concretos |
| 3 | **IEEE Access** | Si se prioriza probabilidad de aceptación sobre factor de impacto |

---

## 13. Fuera del Scope (Trabajo Futuro)

- **Modelos y Re-rankers basados en LLM:** Se excluyen de la evaluación empírica por una decisión de diseño arquitectónico y de control experimental. Los LLMs son recomendadores basados fundamentalmente en contenido (Content-Based) que requieren metadatos textuales ricos (títulos y resúmenes completos). Este benchmark se enfoca intencionalmente en aislar la **señal puramente colaborativa (feedback implícito)**, que es el escenario más común en sistemas reales donde los metadatos son escasos, incompletos o inexistentes (como se evidencia en las colecciones CiteULike-t y BibSonomy). Adicionalmente, el costo computacional y la latencia de inferencia de los LLMs los hacen inviables para catálogos masivos y recomendaciones en tiempo real en entornos con recursos limitados. Así, evaluar el límite del rendimiento de arquitecturas eficientes (LightGCN, SASRec) sobre matrices de interacción pura responde a una pregunta de ingeniería de alto valor: *¿cómo optimizar la recomendación cuando no se dispone de metadatos ricos ni de presupuesto de cómputo para LLMs?*
- **Modelos adicionales (FM, Autoencoders, Difusión):** Se consideran fuera del alcance del presente estudio comparativo para mantener el foco en la comparación de las 5 familias arquitectónicas representativas más extendidas.
- **Datasets masivos (OpenAlex, Semantic Scholar completo):** Quedan como trabajo futuro, requiriendo pipelines de procesamiento distribuido que escapan al alcance de este análisis metodológico.
- **SHAP / Captum sobre embeddings latentes:** Reemplazado por el análisis empírico de sesgo de popularidad y métricas de diversidad, debido a la falta de interpretabilidad semántica de las dimensiones latentes individuales.

---

## 14. Amenazas a la Validez (Threats to Validity)

Se incluye una sección dedicada en el manuscrito para discutir críticamente las limitaciones metodológicas del estudio. Esto ayuda a predecir y responder a posibles objeciones de los revisores.

### 14.1 Validez Interna (Internal Validity)
* **Sesgo de inicialización (Múltiples Semillas):** Para garantizar una reproducibilidad del 100% y mantener los tiempos de cómputo dentro de límites viables (especialmente para modelos neuronales complejos como BERT4Rec y LightGCN), los experimentos se ejecutan bajo una única semilla aleatoria fija (`2024`). Mitigamos este sesgo al evaluar los modelos en múltiples configuraciones de datasets y splits, sirviendo estas variaciones como observaciones de estabilidad independientes.
* **Sesgo de optimización de hiperparámetros:** Para evitar optimizar un modelo más que otro (e.g., sobreajustar LightGCN y dejar ItemKNN por defecto), se decide explícitamente no hacer un estudio de *ablation* propio. En su lugar, se adoptan los hiperparámetros reportados como óptimos en las publicaciones originales de cada modelo y en las implementaciones oficiales de RecBole.

### 14.2 Validez Externa (External Validity)
* **Generalización de dominio:** Al evaluar modelos sobre literatura científica, los resultados podrían estar acotados a una sola plataforma o grupo de usuarios. Mitigamos esta amenaza al utilizar tres datasets de origen independiente: dos de CiteULike (`citeulike-a` y `citeulike-t`) y uno de BibSonomy. BibSonomy cuenta con dinámicas de bookmarking distintas que permiten validar si los patrones de comportamiento de los paradigmas son realmente transversales al dominio académico.

### 14.3 Validez de Constructo (Construct Validity)
* **Métricas utilizadas:** Evaluar solo precisión de ranking (Recall, NDCG) ignora las necesidades reales de un investigador científico (diversidad, descubrimiento de nichos). Mitigamos esto al incorporar métricas de Coverage, Novelty y Diversity en una evaluación multi-criterio tridimensional, junto con un análisis cuantitativo del sesgo de popularidad.

### 14.4 Validez de Conclusión (Conclusion Validity)
* **Significancia de diferencias:** Reportar únicamente métricas absolutas en tablas puede llevar a conclusiones erróneas sobre qué modelo es superior. Mitigamos esto mediante el uso riguroso de pruebas no-paramétricas de significancia (Friedman), tamaño del efecto global (Kendall's W) y comparaciones por pares *post-hoc* (Nemenyi), presentadas visualmente en Diagramas de Diferencia Crítica (CD Diagrams).
