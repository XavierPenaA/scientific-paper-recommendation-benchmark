# 📚 Guía de Lectura de Artículos Científicos Esenciales

Esta guía está diseñada para ayudarte a entender la teoría detrás del estudio comparativo de sistemas de recomendación. Estos artículos son la base científica sobre la que se apoya nuestro paper y son esenciales si vas a discutir este proyecto en una entrevista de doctorado (como en el OIST).

---

## 1. El Origen y el Baseline Clásico

### **Item-Based Collaborative Filtering Recommendation Algorithms**
*   **Autores:** Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl (WWW 2001)
*   **Por qué es clave:** Es el paper fundacional del algoritmo **Item-KNN** (el algoritmo de tu proyecto original de hace 6 años).
*   **Intuición:** En lugar de buscar usuarios similares (lo cual es costoso porque cambian de gustos constantemente), calcula la similitud entre *ítems* basada en cómo los usuarios los han calificado en conjunto. Si a muchos usuarios les gusta la Película A y la Película B, estas son similares.
*   **Qué buscar en la lectura:**
    *   La fórmula de Similitud Coseno Ajustada (Adjusted Cosine Similarity) que soluciona la diferencia de escalas de calificación de los usuarios.
    *   Cómo el cálculo de similitudes se puede hacer *offline* para acelerar la recomendación *online*.

---

## 2. La Era del Deep Learning Colaborativo

### **Neural Collaborative Filtering (NCF)**
*   **Autores:** Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua (WWW 2017)
*   **Por qué es clave:** Intentó jubilar la factorización de matrices tradicional introduciendo redes neuronales.
*   **Intuición:** Reemplaza el producto punto tradicional (que asume relaciones lineales simples entre usuarios e ítems) con un perceptrón multicapa (MLP) que puede aprender interacciones no lineales complejas a partir de sus embeddings latentes. Combina Factorización Matricial Generalizada (GMF) con un MLP en una arquitectura híbrida llamada **NeuMF**.
*   **Qué buscar en la lectura:**
    *   La arquitectura del modelo NeuMF (Figura 2 del paper).
    *   La justificación de por qué el producto punto limita la capacidad de expresar relaciones complejas de similitud.

---

## 3. Recomendación Secuencial (Transformers)

### **Self-Attentive Sequential Recommendation (SASRec)**
*   **Autores:** Wang-Cheng Kang and Julian McAuley (ICDM 2018)
*   **Por qué es clave:** Adaptó los bloques del Transformer de NLP al problema de recomendar el "siguiente ítem" basándose en el historial reciente del usuario.
*   **Intuición:** Modela la recomendación como un problema secuencial. Usa mecanismos de auto-atención (self-attention) unidireccionales (como GPT) para asignar diferentes pesos a los ítems anteriores que ha consumido el usuario, capturando tanto patrones de largo como de corto plazo.
*   **Qué buscar en la lectura:**
    *   La diferencia entre usar RNNs/CNNs (como GRU4Rec) y usar mecanismos de atención para modelar secuencias.
    *   Cómo maneja la escala del historial usando máscaras de causalidad.

### **BERT4Rec: Bidirectional Encoder Representations for Sequential Recommendation**
*   **Autores:** Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang (CIKM 2019)
*   **Por qué es clave:** Introdujo la auto-atención bidireccional en la recomendación.
*   **Intuición:** A diferencia de SASRec (que solo mira el pasado para predecir el futuro), BERT4Rec usa un modelo bidireccional (como BERT) que oculta ciertos ítems del historial del usuario (Cloze task) y entrena al modelo a predecirlos usando el contexto de antes *y* después.
*   **Qué buscar en la lectura:**
    *   La tarea Cloze (Masked Language Modeling adaptado a ítems).
    *   El tradeoff: BERT4Rec suele capturar mejor las dependencias bidireccionales, pero es más difícil de entrenar y evaluar en producción que SASRec.

---

## 4. Redes Neuronales de Grafos (GNNs)

### **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**
*   **Autores:** Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang (SIGIR 2020)
*   **Por qué es clave:** Demostró que las GNNs para recomendación se habían vuelto demasiado complejas e ineficientes sin necesidad.
*   **Intuición:** Modela la interacción usuario-ítem como un grafo bipartito. En lugar de aplicar transformaciones lineales pesadas y activaciones no lineales (como ReLU) en cada capa del grafo (como hacía NGCF), LightGCN solo realiza **propagación lineal de embeddings** de vecinos. Esto lo hace mucho más rápido, fácil de entrenar y preciso.
*   **Qué buscar en la lectura:**
    *   La justificación matemática de por qué las operaciones tradicionales de red neuronal (como Wx + b y activaciones no lineales) no ayudan en el filtrado colaborativo en grafos y cómo eliminarlas mejora el rendimiento.

---

## 5. El Debate Académico de la Reproducibilidad

### **Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches**
*   **Autores:** Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach (RecSys 2019)
*   **Por qué es clave:** El paper más controvertido y citado de RecSys en años recientes.
*   **Intuición:** Los autores intentaron replicar 18 algoritmos de Deep Learning publicados en prestigiosas conferencias de los últimos años. Encontraron que la gran mayoría no lograban superar a algoritmos clásicos y simples (como Item-KNN optimizado o factorización matricial simple) cuando se les evaluaba bajo condiciones justas.
*   **Qué buscar en la lectura:**
    *   Las fallas metodológicas comunes en la investigación de machine learning aplicadas a sistemas de recomendación.
    *   La importancia de optimizar correctamente los hiperparámetros de los algoritmos clásicos (*baselines*).

---

## 6. Explicabilidad y Teoría de Juegos

### **A Unified Approach to Interpreting Model Predictions**
*   **Autores:** Scott M. Lundberg and Su-In Lee (NeurIPS 2017)
*   **Por qué es clave:** El paper que introdujo **SHAP**, la herramienta que usaremos para explicar las recomendaciones.
*   **Intuición:** Propone un método para explicar las predicciones de cualquier modelo de ML (caja negra) asignando a cada variable de entrada un valor de importancia basado en los valores de Shapley de la teoría de juegos cooperativos. Garantiza explicaciones consistentes y localmente fieles.
*   **Qué buscar en la lectura:**
    *   Los tres axiomas matemáticos que un método de explicación de importancia de features debe cumplir: Precisión Local, Ausencia de variables nulas, y Consistencia.
