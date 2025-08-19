
## TITULO Y DESCRIPCION
INTRODUCCION

TelecomX, es una compañía que ofrece servicios de telefonía, internet,vigilancia, con plataformas de streaming Tv y películas online enfrenta una problematica muy común en este tipo de negocios que son las "Cancelaciones" o "Churn", donde el cliente abandona el servicio y se va a la competencia de esta.

La idea principal de este trabajo es la detección temprana de factores que correlacionados con distintas variables del negocio, permitirán a TelecomX, tomar las medidas necesarias para disminuir o evitar la pérdida de clientes.Para ello se buscarán distintas estrategias para fidelizar o mantener al cliente satisfecho con los servicios ofrecidos.
## OBJETIVO GENERAL
Desarrollar un sistema de detección temprana de factores correlacionados con la evasión de clientes (churn) en TelecomX, mediante el análisis de variables del negocio, para implementar estrategias efectivas de retención que permitan disminuir o evitar la pérdida de clientes hacia la competencia.
## OBJETIVOS ESPECÍFICOS
1. **Análisis Exploratorio de Datos**
   - Comprender la estructura y composición del dataset de clientes de TelecomX
   - Identificar las variables más relevantes para el análisis de evasión

2. **Limpieza y Preparación de Datos**
   - Detectar y corregir inconsistencias, valores ausentes y errores de formato en los datos
   - Estandarizar y transformar variables para facilitar el análisis posterior

3. **Análisis Descriptivo del Comportamiento de Clientes**
   - Caracterizar el perfil de clientes que permanecen versus los que cancelan el servicio
   - Identificar patrones de comportamiento asociados con la evasión

4. **Análisis de Correlaciones**
   - Establecer relaciones entre variables categóricas y numéricas con la variable churn
   - Determinar los factores más influyentes en la decisión de cancelación

5. **Generación de Insights Estratégicos**
   - Proporcionar información valiosa para el desarrollo de estrategias de fidelización
   - Facilitar la toma de decisiones para la retención de clientes
## TAREAS SECUENCIALES
###TELECOM X PARTE 2
## **Predicción de Cancelación (Churn) - Pipeline de Machine Learning**

---

##**SECUENCIA COMPLETA DE TAREAS**

### **MISIÓN PRINCIPAL**
Desarrollar modelos predictivos para prever qué clientes tienen mayor probabilidad de cancelar sus servicios, construyendo un pipeline robusto de Machine Learning.

---

## **FASE 1: PREPARACIÓN DE DATOS**

### **1.Extracción del Archivo Tratado**
- **Objetivo**: Cargar el archivo CSV con datos previamente limpiados de la Parte 1
- **Requisito**: Utilizar datos ya corregidos y estandarizados
- **Entregable**: Dataset limpio y organizado

### **2.Eliminación de Columnas Irrelevantes**
- **Objetivo**: Remover columnas que no aportan valor predictivo
- **Foco**: Eliminar identificadores únicos (ej: ID del cliente)
- **Justificación**: Evitar variables que perjudiquen el rendimiento del modelo

### **3.Encoding (Codificación)**
- **Objetivo**: Transformar variables categóricas a formato numérico
- **Método sugerido**: One-hot encoding
- **Herramientas**: get_dummies o OneHotEncoder
- **Propósito**: Hacer datos compatibles con algoritmos de ML

---

## **FASE 2: ANÁLISIS EXPLORATORIO ESPECÍFICO**

### **4.Verificación de la Proporción de Cancelación (Churn)**
- **Objetivo**: Calcular proporción de clientes que cancelaron vs. activos
- **Evaluación**: Detectar desbalance entre clases
- **Herramienta**: value_counts() de pandas
- **Impacto**: Identificar posibles sesgos en modelos

### **5.Normalización o Estandarización (si es necesario)**
- **Evaluación**: Determinar necesidad según modelos a aplicar
- **Modelo que requieren**: KNN
- **Modelos que NO requieren**: Random Forest, 

- **Criterio**: Modelos basados en distancia vs. basados en árboles

### **6.Análisis de Correlación**
- **Objetivo**: Visualizar matriz de correlación entre variables numéricas
- **Foco especial**: Variables con mayor correlación con cancelación
- **Propósito**: Identificar candidatas fuertes para modelo predictivo

### **7.Análisis Dirigido**
- **Investigaciones específicas**:
  - **Tiempo de contrato × Cancelación**
  - **Gasto total × Cancelación**
- **Herramientas**: Boxplots, 
- **Objetivo**: Visualizar patrones y tendencias específicas

---

## **FASE 3: MODELADO**

### **8.Separación de Datos**
- **División**: Entrenamiento y Prueba
- **Proporciones sugeridas**: 
  - 70% entrenamiento / 30% prueba, o
  - 80% entrenamiento / 20% prueba
- **Criterio**: Depende del tamaño del dataset

### **9.Creación de Modelos**
- **Requisito mínimo**: Al menos 2 modelos diferentes
- **Categorías**:
  - **Modelos que requieren normalización**: KNN
  - **Modelos que NO requieren normalización**: Random Forest
- **Justificación**: Comparar sensibilidad a escala de datos

---

## **FASE 4: EVALUACIÓN Y ANÁLISIS**

### **10.Evaluación de los Modelos**
- **Métricas obligatorias**:
  - **Exactitud (Accuracy)**
  - **Precisión (Precision)**
  - **Recall**
  - **F1-score**
  - **Matriz de confusión**

- **Análisis crítico**:
  - ¿Cuál modelo tuvo mejor desempeño?
  - Detección de **Overfitting** o **Underfitting**
  - Identificación de causas y ajustes necesarios

### **11.Análisis de la Importancia de las Variables**
- **Por tipo de modelo**:
  - **KNN**: Influencia de vecinos más cercanos
  - **Random Forest**: Importancia basada en reducción de impureza
  
---

## **FASE 5: ENTREGABLES FINALES**

### **12.Conclusión**
- **Informe detallado** con:
  - **Factores que más influyen** en la cancelación
  - **Variables más relevantes** identificadas
  - **Principales factores** que afectan cancelación de clientes
  - **Estrategias de retención** basadas en resultados

---

## 1.1.IDENTIFICAR Y MANIPULAR COLUMNAS CATEGÓRICAS
## **Análisis Detallado: Identificación y Mapeo de Columnas Categóricas**

Este código representa una implementación sofisticada para la **identificación y preparación del mapeo de variables categóricas** antes de aplicar la codificación one-hot encoding, una etapa crucial en el preprocesamiento de datos para machine learning.

### **Identificación de Variables Categóricas**

La primera línea utiliza el método `select_dtypes(include='object')` para identificar automáticamente todas las columnas que contienen datos de tipo objeto (string), que típicamente corresponden a variables categóricas. Este enfoque es **eficiente y robusto** porque evita la necesidad de especificar manualmente cada columna categórica, reduciendo errores humanos y adaptándose dinámicamente a cambios en la estructura del dataset.

La exclusión de la columna 'cancelacion' es **estratégicamente importante**, ya que esta representa la variable objetivo (target) que queremos predecir. Incluirla en el proceso de encoding podría generar **data leakage** y comprometer la validez del modelo predictivo.

### **Construcción del Diccionario de Mapeo**

El bucle principal construye un diccionario comprehensive que mapea cada valor categórico original a sus futuras columnas binarias. Esta **preparación anticipada** es fundamental porque permite:

1. **Trazabilidad**: Mantener un registro claro de cómo se transformarán los datos
2. **Reversibilidad**: Posibilitar la interpretación de resultados volviendo a las categorías originales
3. **Control de calidad**: Verificar que la transformación se realizará correctamente

### **Manejo del Parámetro drop_first=True**

La sección más compleja del código aborda la lógica del parámetro `drop_first=True` de pandas.get_dummies(). Esta funcionalidad elimina la primera categoría para **evitar multicolinealidad perfecta**, un problema que puede afectar negativamente a modelos lineales como la regresión logística.

Cuando se aplica drop_first=True:
- **La primera categoría se convierte en la "línea de base"**
- **Se representa implícitamente cuando todas las otras columnas binarias son 0**
- **Se reduce la dimensionalidad** del dataset sin pérdida de información

### **Ventajas de esta Implementación**

Este enfoque presenta múltiples **beneficios técnicos**:

**Automatización**: El proceso se adapta automáticamente a cualquier número de variables categóricas y sus respectivos valores únicos, eliminando la necesidad de codificación manual.

**Prevención de errores**: Al crear el mapeo antes de la transformación, se pueden identificar y corregir problemas potenciales como categorías con un solo valor único.

**Optimización de memoria**: El manejo inteligente de drop_first=True reduce el número de columnas resultantes, optimizando el uso de memoria y mejorando la eficiencia computacional.

**Interpretabilidad**: El diccionario resultante facilita la interpretación posterior de los coeficientes del modelo, ya que permite mapear las características transformadas de vuelta a las categorías originales.

### **Consideraciones Técnicas**

El código maneja elegantemente **casos especiales**:
- Columnas con un solo valor único (que no generarían columnas adicionales)
- La lógica de exclusión de la primera categoría
- La creación de mapeos bidireccionales para interpretación

### **Aplicación Práctica en el Contexto Telecom**

En el contexto del análisis de churn de Telecom X, este código prepararía eficientemente variables como:
- **Tipo de contrato**: Month-to-month, One year, Two year
- **Método de pago**: Electronic check, Credit card, Bank transfer, Mailed check
- **Servicios adicionales**: Yes/No para diferentes servicios

La implementación garantiza que estas transformaciones mantengan la **coherencia semántica** mientras optimizan el dataset para algoritmos de machine learning, estableciendo una base sólida para la construcción de modelos predictivos precisos y interpretables.


##1.2.**Análisis: Aplicación de One-Hot Encoding**

### **Transformación Principal**

La línea `pd.get_dummies()` representa el **núcleo de la transformación categórica**, convirtiendo variables cualitativas en representaciones numéricas binarias. Este proceso es **fundamental** para hacer compatibles los datos categóricos con algoritmos de machine learning que solo procesan valores numéricos.

El parámetro `columns=columnas_categoricas` garantiza que **solo las variables categóricas identificadas** previamente sean transformadas, preservando intactas las variables numéricas originales y la variable objetivo 'cancelacion'. Esto mantiene la **integridad estructural** del dataset.

### **Optimización con drop_first=True**

La implementación de `drop_first=True` es **técnicamente estratégica**. Al eliminar la primera categoría de cada variable, se previene la **multicolinealidad perfecta** - un fenómeno donde una columna puede predecirse exactamente a partir de otras columnas. Para modelos lineales como regresión logística, esto evita problemas de **singularidad en matrices** y mejora la estabilidad numérica.

Por ejemplo, si 'tipo_contrato' tiene valores [Month-to-month, One year, Two year], se crearían solo dos columnas binarias: 'tipo_contrato_One year' y 'tipo_contrato_Two year'. Cuando ambas son 0, implícitamente representa 'Month-to-month'.

### **Verificación y Control de Calidad**

Las instrucciones de `print()` implementan un **sistema robusto de validación**:

**Visualización del mapeo**: Permite verificar que la transformación se realizó según lo planificado, facilitando la **trazabilidad** del proceso.

**Inspección del DataFrame resultante**: `df_clientes_encoded.head()` muestra las primeras filas transformadas, permitiendo **validación visual** de la estructura resultante.

**Análisis de metadatos**: `df_clientes_encoded.info()` proporciona información crucial sobre tipos de datos, valores nulos y dimensionalidad del dataset transformado.

### **Impacto en el Pipeline de ML**

Esta transformación **incrementa significativamente la dimensionalidad** del dataset. En el contexto Telecom X, variables como 'metodo_pago' (4 categorías) se convierten en 3 columnas binarias, y 'tipo_internet' (3 categorías) en 2 columnas binarias.

El resultado es un **dataset completamente numérico**, listo para algoritmos de machine learning, manteniendo toda la **información semántica original** mientras optimiza la representación para modelos predictivos. Esta preparación es **esencial** para el éxito de modelos como regresión logística, SVM, y redes neuronales en la predicción de churn.
## 2.VERIFICACIÓN DE LA PROPORCIÓN DE CANCELACIÓN (CHURN)


###2.VERIFICACIÓN DE LA PROPORCIÓN DE CANCELACIÓN (CHURN)

## **Análisis Crítico: Verificación de Proporción y Desbalance de Clases**

### **Propósito Estratégico del Análisis**

Este código implementa una **evaluación fundamental** en cualquier proyecto de machine learning de clasificación: la identificación del balance de clases en la variable objetivo. La verificación del desbalance es **crítica** porque influye directamente en la selección de algoritmos, métricas de evaluación y estrategias de entrenamiento del modelo predictivo.

### **Cálculo de Proporciones**

La línea `df_clientes_encoded['cancelacion'].value_counts(normalize=True) * 100` ejecuta una **transformación elegante y eficiente**. El parámetro `normalize=True` convierte los conteos absolutos en proporciones relativas (valores entre 0 y 1), mientras que la multiplicación por 100 los transforma en **porcentajes interpretables**.

Esta metodología proporciona una **visión inmediata** de la distribución: por ejemplo, si obtenemos 73.46% para "No" (clientes activos) y 26.54% para "Sí" (clientes que cancelaron), comprendemos instantáneamente la naturaleza del problema de negocio.

### **Detección de Desbalance de Clases**

La implementación utiliza un **umbral del 20%** para identificar desbalance significativo, aplicando `proporcion_cancelacion.min()` para encontrar la clase minoritaria. Este enfoque es **técnicamente sólido** porque:

**Flexibilidad**: El umbral puede ajustarse según el contexto del negocio y los requisitos específicos del proyecto.

**Universalidad**: Funciona independientemente de cuál sea la clase minoritaria (cancelaron o no cancelaron).

**Simplicidad**: Proporciona una regla de decisión clara y comprensible para todo el equipo.

### **Implicaciones para el Modelado**

**Si existe desbalance significativo**, surgen **múltiples consideraciones técnicas**:

**Sesgo en las predicciones**: Los modelos tienden a favorecer la clase mayoritaria, prediciendo "No cancelación" con mayor frecuencia, lo que puede ser problemático para identificar clientes en riesgo.

**Métricas engañosas**: La accuracy puede ser alta simplemente prediciendo siempre la clase mayoritaria, sin valor real para el negocio.

**Necesidad de técnicas especializadas**: Requerirá implementar estrategias como SMOTE (sobremuestreo), undersampling (submuestreo), o ajuste de pesos de clase.

### **Contexto Empresarial en Telecom**

En el sector telecomunicaciones, un **desbalance hacia "No cancelación"** es esperado y refleja un negocio saludable. Sin embargo, desde la perspectiva de machine learning, presenta desafíos:

**Costo asimétrico**: No identificar un cliente que va a cancelar (falso negativo) es más costoso que identificar incorrectamente a uno que no va a cancelar (falso positivo).

**Priorización de métricas**: Recall y F1-score se vuelven más relevantes que accuracy para evaluar el rendimiento del modelo.

### **Decisiones Estratégicas Posteriores**

Este análisis **informa decisiones cruciales**:

**Selección de algoritmos**: Modelos como Random Forest con balanceo de clases o XGBoost con parámetros de peso pueden ser más apropiados.

**Estrategias de validación**: Implementar validación cruzada estratificada para mantener proporciones similares en todos los folds.

**Métricas de evaluación**: Priorizar precision, recall y AUC-ROC sobre accuracy simple.

**Técnicas de remuestreo**: Considerar técnicas de balanceo antes del entrenamiento del modelo.

### **Valor Diagnóstico**

Este código no solo identifica un problema potencial, sino que **establece expectativas realistas** sobre el rendimiento del modelo y guía las decisiones de preprocesamiento. Es un ejemplo de **análisis exploratorio dirigido** que combina rigor técnico con comprensión del contexto empresarial, fundamental para el éxito del proyecto de predicción de churn.
## 3.NORMALIZACION DATAFRAME
## **Análisis: Normalización con MinMaxScaler**

### **Selección de Técnica de Normalización**

La implementación utiliza **MinMaxScaler**, una técnica que transforma variables numéricas al rango [0,1] mediante la fórmula: `(valor - mínimo) / (máximo - mínimo)`. Esta elección es **estratégicamente apropiada** para modelos sensibles a escala como regresión logística, KNN y SVM, donde diferencias en magnitud entre variables pueden sesgar los resultados.

### **Identificación Automática de Variables**

El código emplea `select_dtypes(include=['int64', 'float64'])` para identificar automáticamente columnas numéricas, excluyendo implícitamente variables categóricas codificadas (que ya están en escala 0-1) y la variable objetivo. Esta **automatización inteligente** previene errores manuales y se adapta dinámicamente a cambios en la estructura del dataset.

### **Ventajas del MinMaxScaler**

**Preservación de distribuciones**: Mantiene la forma original de las distribuciones, solo cambiando la escala, lo que es crucial para variables como 'tenencia' o 'cuota_mensual' que pueden tener distribuciones asimétricas importantes.

**Interpretabilidad**: Los valores normalizados entre 0-1 son **intuitivamente comprensibles**: 0 representa el valor mínimo histórico y 1 el máximo.

**Robustez**: Funciona efectivamente con variables que tienen diferentes unidades de medida (meses vs. dólares vs. conteos).

### **Impacto en el Contexto Telecom**

En el dataset de Telecom X, esta normalización es **particularmente valiosa**:
- **Tenencia** (0-70+ meses) y **cuota_total** ($0-8000+) tienen escalas completamente diferentes
- **Cuentas_diarias** (0-5) requiere equiparación con otras métricas monetarias
- La normalización garantiza que ninguna variable domine el modelo por su magnitud

### **Consideraciones Técnicas**

**Aplicación in-place**: La modificación directa de `df_clientes_encoded[columnas_numericas]` es **eficiente en memoria** pero requiere precaución para mantener trazabilidad.

**Fit_transform**: El uso de `fit_transform()` calcula estadísticas (min/max) y aplica la transformación simultáneamente, apropiado para el conjunto de entrenamiento.

**Persistencia del scaler**: Aunque no se muestra, guardar el objeto `scaler` es crucial para aplicar la misma transformación a datos futuros de predicción.

### **Preparación para Modelado**

Esta normalización **optimiza el rendimiento** de algoritmos como regresión logística (donde coeficientes son comparables) y KNN (donde distancias euclidianas son meaningfully calculadas). El resultado es un dataset **uniformemente escalado** que maximiza la capacidad predictiva de modelos sensibles a magnitud, estableciendo una base sólida para la fase de entrenamiento y evaluación de modelos.
Aquí tienes la información estructurada en formato **Markdown**, con mejor presentación y legibilidad para informes o documentación técnica:

```markdown
## 📋 Resumen de las Columnas del Dataset

| #  | Columna                                      | Non-Null Count | Dtype    |
|----|----------------------------------------------|----------------|----------|
| 0  | cancelacion                                  | 7032 non-null  | float64  |
| 1  | adulto_mayor                                 | 7032 non-null  | float64  |
| 2  | tiene_pareja                                 | 7032 non-null  | float64  |
| 3  | tiene_dependientes                           | 7032 non-null  | float64  |
| 4  | tenencia                                     | 7032 non-null  | float64  |
| 5  | tiene_telefono                               | 7032 non-null  | float64  |
| 6  | facturacion_electronica                      | 7032 non-null  | float64  |
| 7  | cuota_mensual                                | 7032 non-null  | float64  |
| 8  | cuota_total                                  | 7032 non-null  | float64  |
| 9  | cuentas_diarias                              | 7032 non-null  | float64  |
| 10 | genero_Male                                  | 7032 non-null  | bool     |
| 11 | lineas_multiples_No phone service            | 7032 non-null  | bool     |
| 12 | lineas_multiples_Yes                         | 7032 non-null  | bool     |
| 13 | tipo_internet_Fiber optic                    | 7032 non-null  | bool     |
| 14 | tipo_internet_No                             | 7032 non-null  | bool     |
| 15 | seguridad_online_No internet service         | 7032 non-null  | bool     |
| 16 | seguridad_online_Yes                         | 7032 non-null  | bool     |
| 17 | respaldo_online_No internet service          | 7032 non-null  | bool     |
| 18 | respaldo_online_Yes                          | 7032 non-null  | bool     |
| 19 | proteccion_dispositivo_No internet service   | 7032 non-null  | bool     |
| 20 | proteccion_dispositivo_Yes                   | 7032 non-null  | bool     |
| 21 | soporte_tecnico_No internet service          | 7032 non-null  | bool     |
| 22 | soporte_tecnico_Yes                          | 7032 non-null  | bool     |
| 23 | streaming_tv_No internet service             | 7032 non-null  | bool     |
| 24 | streaming_tv_Yes                             | 7032 non-null  | bool     |
| 25 | streaming_peliculas_No internet service      | 7032 non-null  | bool     |
| 26 | streaming_peliculas_Yes                      | 7032 non-null  | bool     |
| 27 | contrato_One year                            | 7032 non-null  | bool     |
| 28 | contrato_Two year                            | 7032 non-null  | bool     |
| 29 | metodo_pago_Credit card (automatic)          | 7032 non-null  | bool     |
| 30 | metodo_pago_Electronic check                 | 7032 non-null  | bool     |
| 31 | metodo_pago_Mailed check                     | 7032 non-null  | bool     |

> ✅ **Total de filas:** 7032  
> ✅ **Sin valores nulos** en ninguna columna  
> ✅ **Tipos de datos:**  
> - Variables binarias codificadas como `bool` (dummies)  
> - Variables numéricas (`float64`) usadas como binarias (0/1) para `cancelacion`, `adulto_mayor`, etc.
```

---

### 🔍 Notas importantes:
- Las variables categóricas han sido **codificadas con One-Hot Encoding** (por ejemplo, `tipo_internet`, `metodo_pago`, etc.).
- La ausencia de valores nulos indica que el **dataset está limpio y listo para modelado**.
- La variable objetivo es `cancelacion`:
  - `1` = Cliente canceló
  - `0` = Cliente se retuvo

Este resumen es ideal para incluir en un informe de exploración de datos (EDA) o documentación de proyecto de ciencia de datos.
## 4.CORRELACIÓN Y SELECCIÓN DE VARIABLES
## **Análisis: Matriz de Correlación y Selección de Variables**

### **Transformación de Variable Objetivo**

El código implementa una **conversión estratégica** de la variable categórica 'cancelacion' a formato numérico (0/1), esencial para calcular correlaciones de Pearson. La inclusión de `np.nan` para valores vacíos demuestra **robustez** en el manejo de datos incompletos, seguido de una limpieza apropiada con `dropna()`.

### **Preparación de Datos para Correlación**

La selección de `columnas_numericas_y_booleanas` usando `select_dtypes(include=[np.number, 'bool'])` es **técnicamente precisa**, ya que las correlaciones de Pearson requieren variables numéricas. Esta identificación automática incluye tanto variables numéricas originales (normalizadas) como las columnas binarias creadas por one-hot encoding.

### **Visualización Optimizada**

El heatmap con `figsize=(24, 12)` y parámetros ajustados aborda el **desafío de dimensionalidad** típico después del one-hot encoding. La configuración `annot=True` con `fmt=".1f"` proporciona valores numéricos precisos, mientras que `cmap='coolwarm'` facilita la **identificación visual** de correlaciones positivas (rojas) y negativas (azules).

### **Análisis Dirigido de Predictores**

La línea final `matrix_correlacion['cancelacion_numerica'].sort_values(ascending=False)` extrae específicamente las correlaciones con la variable objetivo, ordenadas de mayor a menor. Este **análisis dirigido** identifica inmediatamente las variables con mayor poder predictivo, tanto positivo como negativo.

### **Valor Estratégico**

Este análisis revela **insights cruciales** para el modelado:
- **Variables fuertemente correlacionadas** con cancelación se convierten en candidatas prioritarias
- **Correlaciones negativas** identifican factores de retención
- **Correlaciones cercanas a cero** sugieren variables poco informativas

La visualización completa permite detectar **multicolinealidad** entre predictores, informando decisiones sobre selección de características y evitando redundancia en el modelo. Este paso es **fundamental** para optimizar el rendimiento predictivo y la interpretabilidad del modelo final de churn.

Aquí tienes la información presentada en formato **Markdown**, con una estructura clara y profesional para incluir en un informe de análisis de datos:

```markdown
## 🔗 Correlación con la Variable `cancelacion`

La siguiente tabla muestra las correlaciones (coeficientes de Pearson) entre la variable objetivo `cancelacion` y el resto de variables del dataset, ordenadas de mayor a menor impacto:

| Variable                                      | Correlación con `cancelacion` |
|----------------------------------------------|-------------------------------|
| cancelacion                                  | 1.000000                      |
| cancelacion_numerica                         | 1.000000                      |
| tipo_internet_Fiber optic                    | 0.307463 🚨                   |
| metodo_pago_Electronic check                 | 0.301455 🚨                   |
| cuota_mensual                                | 0.192858 ✅                   |
| cuentas_diarias                              | 0.192857 ✅                   |
| facturacion_electronica                      | 0.191454 ✅                   |
| adulto_mayor                                 | 0.150541 ✅                   |
| streaming_tv_Yes                             | 0.063254                      |
| streaming_peliculas_Yes                      | 0.060860                      |
| lineas_multiples_Yes                         | 0.040033                      |
| tiene_telefono                               | 0.011691                      |
| genero_Male                                  | -0.008545                     |
| lineas_multiples_No phone service            | -0.011691                     |
| proteccion_dispositivo_Yes                   | -0.066193                     |
| respaldo_online_Yes                          | -0.082307                     |
| metodo_pago_Mailed check                     | -0.090773                     |
| metodo_pago_Credit card (automatic)          | -0.134687                     |
| tiene_pareja                                 | -0.149982                     |
| tiene_dependientes                           | -0.163128                     |
| soporte_tecnico_Yes                          | -0.164716                     |
| seguridad_online_Yes                         | -0.171270                     |
| contrato_One year                            | -0.178225                     |
| cuota_total                                  | -0.199484                     |
| tipo_internet_No                             | -0.227578 💚                 |
| streaming_tv_No internet service             | -0.227578 💚                 |
| seguridad_online_No internet service         | -0.227578 💚                 |
| respaldo_online_No internet service          | -0.227578 💚                 |
| proteccion_dispositivo_No internet service   | -0.227578 💚                 |
| streaming_peliculas_No internet service      | -0.227578 💚                 |
| soporte_tecnico_No internet service          | -0.227578 💚                 |
| contrato_Two year                            | -0.301552 💚                 |
| tenencia                                     | -0.354049 💚                 |

> 🚨 = Correlación **positiva alta**: asociado a **mayor probabilidad de cancelación**  
> ✅ = Correlación **positiva moderada**  
> 💚 = Correlación **negativa fuerte**: asociado a **menor probabilidad de cancelación (mayor retención)**

4.1.COEFICIENTE DE CORRELACIÓN DE PEARSON ENTRE CADA VARIABLE Y 'CANCELACION_NUMERICA'
Estos resultados muestran el coeficiente de correlación de Pearson entre cada variable y la variable objetivo 'cancelacion_numerica' (donde 1 representa cancelación y 0 no cancelación).

Coeficiente de Correlación: Este valor varía entre -1 y 1.

Un valor cercano a 1 indica una fuerte correlación positiva: a medida que el valor de la variable aumenta, la probabilidad de cancelación también tiende a aumentar.
Un valor cercano a -1 indica una fuerte correlación negativa: a medida que el valor de la variable aumenta, la probabilidad de cancelación tiende a disminuir.
Un valor cercano a 0 indica una correlación débil o nula.
Análisis de las variables con mayor correlación (positiva):

tipo_internet_Fiber optic (0.307):

Tener servicio de fibra óptica tiene una correlación positiva moderada con la cancelación.
Esto sugiere que los clientes con este tipo de internet son más propensos a cancelar.
metodo_pago_Electronic check (0.301):

Utilizar el pago electrónico también muestra una correlación positiva moderada con la cancelación.
Los clientes que pagan de esta manera podrían tener una mayor tendencia a irse.
cuota_mensual (0.193), cuentas_diarias (0.193), facturacion_electronica (0.191):

Estas variables relacionadas con los costos y la facturación electrónica tienen correlaciones positivas más débiles, pero aún sugieren que costos mensuales más altos o la facturación electrónica podrían estar asociados con una mayor probabilidad de cancelación.
adulto_mayor (0.151):

Ser un adulto mayor tiene una correlación positiva débil con la cancelación.
Análisis de las variables con mayor correlación (negativa):

tenencia (-0.354):

La antigüedad o "tenencia" del cliente tiene la correlación negativa más fuerte con la cancelación.
Esto es intuitivo: cuanto más tiempo lleva un cliente, menos probable es que cancele.
contrato_Two year (-0.302):

Tener un contrato de dos años tiene una correlación negativa moderada. Los clientes con contratos más largos son menos propensos a cancelar, ya que están comprometidos por más tiempo.
tipo_internet_No (-0.228), streaming_tv_No internet service (-0.228), etc.: - Varias variables indicando la ausencia de ciertos servicios (como no tener servicio de internet, o no tener servicios de streaming si no tienen internet) tienen correlaciones negativas.

Esto sugiere que los clientes con menos servicios o sin internet son menos propensos a cancelar (quizás porque tienen menos opciones o un servicio básico más estable).
cuota_total (-0.199):

La cuota total acumulada tiene una correlación negativa débil. Esto puede parecer contradictorio con la cuota mensual, pero la cuota total está fuertemente correlacionada con la antigüedad (cuanto más tiempo, mayor la cuota total), lo que refuerza la idea de que los clientes más antiguos son menos propensos a cancelar.
En conclusión:

Las variables con mayor impacto en la predicción de la cancelación, basándonos en la correlación, parecen ser la antigüedad del cliente (tenencia), el tipo de contrato (dos años vs otros), el tipo de internet (fibra óptica) y el método de pago (cheque electrónico).

Las cuotas mensuales y la facturación electrónica también muestran una relación positiva, aunque más débil.

Las variables con correlaciones cercanas a cero (como el género o si tiene teléfono) probablemente tienen menos poder predictivo individualmente.

Es importante recordar que la correlación solo mide relaciones lineales y no implica causalidad.

Aun así, estas variables son buenos candidatos para incluir en un modelo predictivo de cancelación.
##5.ANÁLISIS DIRIGIDO
## **Análisis: Visualización de Relaciones Clave con Cancelación**

### **Transformación y Preparación**

El código convierte la variable 'cancelacion' a tipo entero, **estandarizando** la representación numérica para facilitar la visualización. Esta conversión es **técnicamente necesaria** para que seaborn procese correctamente la variable como categórica binaria en los gráficos.

### **Análisis de Contrato vs. Cancelación**

La primera visualización utiliza `countplot` para examinar la relación entre 'contrato_One year' (variable binaria post-encoding) y cancelación. Esta representación es **parcialmente limitada** porque solo muestra una dimensión del tipo de contrato original (One year vs. otros), perdiendo información sobre Month-to-month y Two year. Un enfoque más **completo** requeriría análisis separados para cada variable de contrato codificada.

### **Distribución Monetaria por Cancelación**

El boxplot de 'cuota_total' vs. 'cancelacion' proporciona **insights valiosos** sobre patrones monetarios. Esta visualización revela:
- **Medianas comparativas** entre clientes que cancelan vs. retienen
- **Dispersión de gastos** en cada grupo
- **Valores atípicos** que podrían indicar segmentos especiales de clientes

### **Limitaciones Técnicas**

El análisis presenta **oportunidades de mejora**:
- **Enfoque fragmentado** en variables one-hot encoded limita la visión holística
- **Ausencia de análisis de tenencia**, una variable crítica identificada previamente
- **Visualizaciones aisladas** que no exploran interacciones entre variables

### **Valor Estratégico**

A pesar de las limitaciones, estos gráficos proporcionan **validación visual** de patrones identificados en el análisis correlacional. El boxplot especialmente es **crucial** para entender si clientes que cancelan tienen características monetarias distintivas, informando estrategias de segmentación y retención.
Aquí tienes la información presentada en formato **Markdown**, con un estilo claro y profesional, ideal para informes de análisis de datos:


## 📊 Distribución de la Variable `cancelacion`

| cancelacion | Cantidad de Clientes | Porcentaje (%) |
|-------------|------------------------|----------------|
| 0 (No cancela) | 5,163                 | 73.42%         |
| 1 (Cancela)    | 1,869                 | 26.58%         |
| **Total**      | **7,032**             | **100.00%**    |

> 🔎 **Nota:**  
> - `0` = Cliente **retenido**  
> - `1` = Cliente que **canceló el servicio**

### ✅ Interpretación
La distribución muestra que:
- El **73.42%** de los clientes **no han cancelado** el servicio.
- El **26.58%** ha cancelado, lo que indica una **tasa de churn moderada**.
- No se observa un **desequilibrio extremo** en la variable objetivo, lo cual es favorable para modelado predictivo sin necesidad inmediata de técnicas de balanceo (como SMOTE o submuestreo).

💡 **Implicación para modelos de Machine Learning:**  
Este nivel de balance permite entrenar modelos con buena capacidad de generalización, aunque aún puede ser útil evaluar métricas como **precisión, recall y F1-score**, especialmente para la clase minoritaria (clientes que cancelan).

##5.1.DISTRIBUCIÓN DE CANCELACIÓN POR TIPO DE CONTRATO (AÑO UNO):
- El gráfico de barras (countplot) muestra cuántos clientes cancelaron (barras verdes, 'cancelacion' = 1) y cuántos no cancelaron (barras azules, 'cancelacion' = 0) para cada categoría de la variable contrato_One year (Si tienen contrato de un año o no).

-Se puede observar que la mayoría de los clientes que no tienen un contrato de un año (contrato_One year = False) no cancelan (barra azul alta). 
- Sin embargo, el número de cancelaciones (barra verde) es significativamente mayor en este grupo en comparación con los clientes que sí tienen un contrato de un año (contrato_One year = True).

- Para los clientes con un contrato de un año (contrato_One year = True), la gran mayoría no cancela, y el número de cancelaciones es muy bajo.

**Conclusión de este gráfico:** 
- Los clientes con contratos de un año son mucho menos propensos a cancelar que aquellos con otros tipos de contrato (como mes a mes, que se representa en el grupo "False" después del one-hot encoding y drop_first=True). 
- Esto refuerza la idea de que los contratos a largo plazo reducen la tasa de cancelación.

**Distribución de Cuota Total por Cancelación (Boxplot):**

- Este gráfico de caja (boxplot) muestra la distribución de la **cuota_total** (el cargo total acumulado) para los clientes que no cancelaron (0) y los que sí cancelaron (1).
- La "caja" central representa el rango intercuartílico (donde se encuentra el 50% medio de los datos). 
- La línea dentro de la caja es la mediana. Las "líneas" (whiskers) se extienden para mostrar el rango de los datos, excluyendo los valores atípicos (puntos individuales).
- Observamos que la cuota_total para los clientes que no cancelaron (0) tiene una mediana y un rango intercuartílico más altos en comparación con los clientes que sí cancelaron (1). 
- Los puntos individuales por encima de las líneas son valores atípicos.

**Conclusión de este gráfico:**
- Los clientes que cancelaron tienden a tener una **cuota total** acumulada menor en promedio en comparación con los clientes que se quedan. 
- Esto podría estar relacionado con el hecho de que los clientes que se quedan generalmente tienen una antigüedad ("**tenencia**") mayor, lo que resulta en una **cuota total** más alta con el tiempo. 
- Esto concuerda con la correlación negativa observada anteriormente entre la **tenencia y la cancelación**.

**En resumen**, estas visualizaciones confirman hallazgos importantes del análisis de correlación: los contratos de un año están asociados con una menor cancelación, y los clientes que cancelan tienden a tener una cuota total acumulada más baja, probablemente debido a una menor antigüedad.

## 6.MODELADO PREDICTIVO

##6.1.SEPARACIÓN DE DATOS

Aquí tienes la información presentada en formato **Markdown**, con un estilo claro y profesional, ideal para incluir en un informe de modelado de machine learning:


## 📏 División del Conjunto de Datos

| Conjunto               | Forma (Shape)       | Descripción                            |
|------------------------|---------------------|----------------------------------------|
| `X_train`              | (5625, 32)          | 5,625 muestras de entrenamiento con 32 características cada una. |
| `X_test`               | (1407, 32)          | 1,407 muestras de prueba con 32 características cada una. |
| `y_train`              | (5625,)             | Etiquetas de entrenamiento (variable objetivo). |
| `y_test`               | (1407,)             | Etiquetas de prueba (variable objetivo). |

### 🔍 Detalles:
- **Proporción de división:** 80% entrenamiento – 20% prueba.
- **Número total de muestras:** 7,032.
- **Número de características (variables predictoras):** 32.
- **Variable objetivo:** `cancelacion` (binaria: 0 = no cancela, 1 = cancela).

✅ **Conclusión:**  
La división es adecuada para entrenar y evaluar modelos de clasificación.  
El tamaño del conjunto de prueba (1,407 muestras) es suficiente para obtener estimaciones confiables del rendimiento del modelo.

Los resultados que se muestran corresponden a la división de los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split de Scikit-learn.

**Forma del conjunto de entrenamiento (X_train):** (5625, 32)

- Esto indica que el conjunto de datos de entrenamiento para las características (X_train) tiene 5625 filas y 32 columnas.
- Las 5625 filas representan el número de ejemplos de clientes que se utilizarán para entrenar el modelo.
- Las 32 columnas representan el número de características o variables (después de la codificación One-Hot y la normalización) que se usarán para predecir la cancelación. 
- La variable objetivo ('cancelacion') ha sido excluida de este conjunto.

**Forma del conjunto de prueba (X_test):** (1407, 32)

-Esto indica que el conjunto de datos de prueba para las características (X_test) tiene 1407 filas y 32 columnas.
- Las 1407 filas representan el número de ejemplos de clientes que se utilizarán para evaluar el rendimiento del modelo después de haber sido entrenado. 
- Estos datos no fueron vistos por el modelo durante el entrenamiento.
- Las 32 columnas son las mismas características que en el conjunto de entrenamiento.

**Forma de las etiquetas de entrenamiento (y_train):** (5625,)

- Esto indica que las etiquetas de la variable objetivo (y_train, que es 'cancelacion') para el conjunto de entrenamiento tienen 5625 filas.
- Esta es una serie o arreglo unidimensional donde cada elemento corresponde a la etiqueta de cancelación (0 o 1) para cada cliente en el conjunto de entrenamiento X_train.

**Forma de las etiquetas de prueba (y_test):** (1407,)

- Esto indica que las etiquetas de la variable objetivo (y_test, 'cancelacion') para el conjunto de prueba tienen 1407 filas.
- Similar a y_train, cada elemento es la etiqueta de cancelación real para cada cliente en el conjunto de prueba X_test.

**En resumen:** 
- Estos resultados muestran que dividiste tu conjunto de datos original en un 80% para entrenamiento (5625 muestras) y un 20% para prueba (1407 muestras), manteniendo las 32 características para ambos conjuntos, además de las etiquetas de cancelación correspondientes. 
- Esta división es un paso estándar para preparar los datos para el modelado predictivo y evaluar el modelo de manera justa.

## 7.CREACIÓN DE MODELOS

##7.1. MODELO CON NORMALIZACIÓN

### 7.1.1. K-Nearest Neighbors (KNN)


## 📊 Evaluación del Modelo: K-Nearest Neighbors (KNN)


## Evaluación del modelo K-Nearest Neighbors (KNN):

**Accuracy:** 0.7647

### Classification Report:

|    |   precision |   recall |   f1-score |   support |
|----|-------------|----------|------------|-----------|
| 0  |        0.83 |     0.85 |       0.84 |      1033 |
| 1  |        0.56 |     0.52 |       0.54 |       374 |
|    |             |          |            |           |
| accuracy |         |          |       0.76 |      1407 |
| macro avg |     0.70 |     0.69 |       0.69 |      1407 |
| weighted avg |  0.76 |     0.76 |       0.76 |      1407 |


Exactitud (Accuracy): 0.7647

Esto significa que el modelo KNN clasificó correctamente aproximadamente el 76.47% del total de clientes en el conjunto de prueba.
Es una medida general del rendimiento del modelo.
Reporte de Clasificación:

Este reporte desglosa las métricas por cada clase (0: No Cancelación, 1: Cancelación).
Precision (Clase 0 - No Cancelación):* 0.83
De todos los clientes que el modelo predijo que no cancelarían, el 83% realmente no cancelaron.
Recall (Clase 0 - No Cancelación): 0.85

De todos los clientes que no cancelaron en la realidad, el modelo identificó Elemento de la lista correctamente al 85%.

Esto indica que el modelo es bueno para identificar a los clientes que se quedarán.
F1-score (Clase 0 - No Cancelación): 0.84

Es un promedio ponderado de la precisión y el recall para la clase "No Cancelación".
Un F1-score alto (0.84) confirma el buen rendimiento del modelo para esta clase.
Precision (Clase 1 - Cancelación): 0.56

De todos los clientes que el modelo predijo que sí cancelarían, el 56% realmente cancelaron.
Esto significa que hay un 44% de "falsos positivos" (clientes predichos como cancelados que no lo hicieron).
Recall (Clase 1 - Cancelación): 0.52

De todos los clientes que sí cancelaron en la realidad, el modelo identificó correctamente solo al 52%.
Esto indica que hay un 48% de "falsos negativos" (clientes que cancelaron pero el modelo predijo que no lo harían).
Al igual que Random Forest, este es un punto débil del modelo para identificar a la clase minoritaria.
F1-score (Clase 1 - Cancelación): 0.54

El F1-score para la clase "Cancelación" (0.54) es significativamente más bajo que para la clase "No Cancelación", reflejando el desafío del modelo para manejar esta clase minoritaria.
Confusion Matrix: Esta matriz muestra el número de predicciones correctas e incorrectas.

[[882 151]

882 (Verdaderos Negativos): Clientes que no cancelaron y el modelo predijo correctamente que no cancelarían. 151 (Falsos Positivos): Clientes que no cancelaron pero el modelo predijo incorrectamente que cancelarían.

[180 194]]

180 (Falsos Negativos): Clientes que sí cancelaron pero el modelo predijo incorrectamente que no cancelarían. 194 (Verdaderos Positivos): Clientes que sí cancelaron y el modelo predijo correctamente que cancelarían.

En resumen:

El modelo KNN tiene una precisión general aceptable (76.47%). Similar al Random Forest, su rendimiento es mejor para predecir la clase mayoritaria (No Cancelación) que para la clase minoritaria (Cancelación).
Tiene una precisión y recall bajos para la clase 'Yes', lo que significa que a menudo predice incorrectamente que un cliente no cancelará cuando en realidad sí lo hace (muchos falsos negativos), aunque ligeramente menos que el Random Forest en este caso.
Esto sugiere que el modelo KNN con la configuración actual también puede no ser el mejor para identificar a todos los clientes que realmente cancelarán.

##7.2.MODELO SIN NORMALIZACIÓN


## Evaluación del modelo Random Forest:

**Accuracy:** 0.7946

### Classification Report:

|                    |   precision |   recall |   f1-score |   support |
|--------------------|-------------|----------|------------|-----------|
| 0                  |        0.83 |     0.90 |       0.87 |      1033 |
| 1                  |        0.65 |     0.50 |       0.56 |       374 |
|                    |             |          |            |           |
| **accuracy**       |             |          |       0.79 |      1407 |
| **macro avg**      |        0.74 |     0.70 |       0.71 |      1407 |
| **weighted avg**   |        0.78 |     0.79 |       0.79 |      1407 |

### Confusion Matrix:


[[931 102]
 [187 187]]

**Evaluación del modelo Random Forest:**

**Accuracy (Exactitud):** 0.7946
- Esto significa que el modelo Random Forest clasificó correctamente aproximadamente el 79.46% del total de clientes en el conjunto de prueba. 
- Es una medida general del rendimiento del modelo.

**Classification Report:**
 Este reporte desglosa las métricas por cada clase (0: No Cancelación, 1: Cancelación).

**Precision (Clase 0 - No Cancelación):** 0.83
De todos los clientes que el modelo predijo que no cancelarían, el 83% realmente no cancelaron.

**Recall (Clase 0 - No Cancelación):** 0.90
- De todos los clientes que no cancelaron en la realidad, el modelo identificó correctamente al 90%. 
- Esto indica que el modelo es muy bueno para identificar a los clientes que se quedarán.

**F1-score (Clase 0 - No Cancelación):** 0.87
- Es un promedio ponderado de la precisión y el recall para la clase "No Cancelación". 
- Un F1-score alto (0.87) confirma el buen rendimiento del modelo para esta clase.

**Precision (Clase 1 - Cancelación):** 0.65
- De todos los clientes que el modelo predijo que sí cancelarían, el 65% realmente cancelaron. 
- Esto significa que hay un 35% de "falsos positivos" (clientes predichos como cancelados que no lo hicieron).

**Recall (Clase 1 - Cancelación):** 0.50
- De todos los clientes que sí cancelaron en la realidad, el modelo identificó correctamente solo al 50%. 
- Esto indica que hay un 50% de "falsos negativos" (clientes que cancelaron pero el modelo predijo que no lo harían). 
- Este es un punto débil del modelo para identificar a la clase minoritaria.

**F1-score (Clase 1 - Cancelación):** 0.56
- El F1-score para la clase "Cancelación" (0.56) es significativamente más bajo que para la clase "No Cancelación", reflejando el desafío del modelo para manejar esta clase minoritaria.

**Confusion Matrix:**
- Esta matriz muestra el número de predicciones correctas e incorrectas.

[[931 102]

**931 (Verdaderos Negativos):** Clientes que no cancelaron y el modelo predijo correctamente que no cancelarían.
**102 (Falsos Positivos):** Clientes que no cancelaron pero el modelo predijo incorrectamente que cancelarían.

[187 187]]

**187 (Falsos Negativos):** Clientes que sí cancelaron pero el modelo predijo incorrectamente que no cancelarían.
**187 (Verdaderos Positivos):** Clientes que sí cancelaron y el modelo predijo correctamente que cancelarían.

**En resumen:** 
- El modelo Random Forest tiene una buena exactitud general y es muy efectivo prediciendo a los clientes que no cancelarán. 
- Sin embargo, tiene dificultades para identificar a los clientes que sí cancelarán, como lo indican su menor precisión y recall (especialmente el recall del 50%) para la clase **cancelación**. 
- Esto significa que el modelo pasa por alto a una parte significativa de los clientes que finalmente cancelan.


###7.2.2. EVALUACIÓN DEL MODELO K-NEAREST NEIGHBORS (KNN)


## Evaluación del modelo K-Nearest Neighbors (KNN):

**Exactitud (Accuracy):** 0.7647

### Reporte de Clasificación:

|                    |   precision |   recall |   f1-score |   support |
|--------------------|-------------|----------|------------|-----------|
| 0                  |        0.83 |     0.85 |       0.84 |      1033 |
| 1                  |        0.56 |     0.52 |       0.54 |       374 |
|                    |             |          |            |           |
| **accuracy**       |             |          |       0.76 |      1407 |
| **macro avg**      |        0.70 |     0.69 |       0.69 |      1407 |
| **weighted avg**   |        0.76 |     0.76 |       0.76 |      1407 |

### Matriz de Confusión:

[[882 151]
 [180 194]]

###Evaluación del modelo Random Forest:
Exactitud (Accuracy): 0.7945984363894811
##Reporte de Clasificación:
              precision    recall  f1-score   support

           0       0.83      0.90      0.87      1033
           1       0.65      0.50      0.56       374

    accuracy                           0.79      1407
   macro avg       0.74      0.70      0.71      1407
weighted avg       0.78      0.79      0.79      1407

Matriz de Confusión:
[[931 102]
 [187 187]] 


##7.3.COMPARACIÓN Y ANÁLISIS CRÍTICO DE MODELOS

| MÉTRICA                                  | KNN   | RANDOM FOREST | OBSERVACIONES                                                                 |
|------------------------------------------|-------|---------------|-------------------------------------------------------------------------------|
| Exactitud (Accuracy)                     | 0.76  | 0.79          | Random Forest tiene una precisión general ligeramente mayor que KNN.          |
| Precision (Clase 'Yes' - cancelación)    | 0.56  | 0.65          | Random Forest es mejor identificando correctamente a los clientes que cancelan entre todas sus predicciones 'Yes'. |
| Recall (Clase 'Yes' - cancelación)       | 0.52  | 0.50          | KNN identifica ligeramente más clientes que cancelan (Verdaderos Positivos) de todos los que realmente cancelan. |
| F1-score (Clase 'Yes' - cancelación)     | 0.54  | 0.56          | Random Forest tiene un mejor equilibrio entre precisión y recall para la clase 'Yes'. |
| Precision (Clase 'No' - no cancelación)  | 0.83  | 0.83          | Ambos modelos tienen una precisión similar para predecir clientes que no cancelan. |
| Recall (Clase 'No' - no cancelación)     | 0.85  | 0.90          | Random Forest es significativamente mejor identificando clientes que no cancelan de todos los que no cancelan. |
| F1-score (Clase 'No' - no cancelación)   | 0.84  | 0.87          | Random Forest tiene un mejor equilibrio para la clase 'No'.                   |
| Falsos Positivos (FP)                    | 151   | 102           | Random Forest comete menos errores al predecir 'Yes' cuando la realidad es 'No'. |
| Falsos Negativos (FN)                    | 180   | 187           | KNN comete ligeramente menos errores al predecir 'No' cuando la realidad es 'Yes'. |
| Verdaderos Positivos (TP)                | 194   | 187           | KNN identifica ligeramente más clientes que cancelan correctamente.           |
| Verdaderos Negativos (TN)                | 882   | 931           | Random Forest identifica correctamente a más clientes que no cancelan.        |

Basado en el cuadro comparativo y las métricas de rendimiento para predecir la cancelación de clientes ('Yes'), aquí tienes un análisis crítico y una conclusión:

Análisis Crítico:

Exactitud General (Accuracy):

El modelo Random Forest tiene una exactitud general ligeramente superior (0.79 vs 0.76), lo que indica que clasifica correctamente una proporción mayor de clientes en general.
Enfoque en la Cancelación (Clase 'Yes'): Aquí es donde radica la diferencia crucial para este problema.

Precision (Clase 'Yes'):

Random Forest tiene una precisión notablemente mayor (0.65 vs 0.56).
Esto significa que, cuando el modelo Random Forest predice que un cliente va a cancelar, es más probable que esa predicción sea correcta en comparación con el modelo KNN.
Esto es valioso para no invertir recursos de retención en clientes que no planean irse (reduciendo falsos positivos).
Recall (Clase 'Yes'):

El modelo KNN tiene un recall ligeramente superior (0.52 vs 0.50).
Esto indica que el modelo KNN es marginalmente mejor identificando a los clientes que realmente van a cancelar de todos los clientes que cancelan.
Un recall bajo en la clase positiva (cancelación) significa que el modelo tiene muchos "falsos negativos" (clientes que cancelan pero el modelo predice que no lo harán), lo cual es un problema significativo si el objetivo es identificar a todos los clientes propensos a la cancelación para aplicar estrategias de retención.
F1-Score (Clase 'Yes'):

El F1-score, que equilibra precisión y recall, es ligeramente mejor para Random Forest (0.56 vs 0.54), lo que sugiere que, en general, tiene un mejor balance para la clase de interés (cancelación).
Rendimiento en la No Cancelación (Clase 'No'):

Ambos modelos son muy buenos prediciendo a los clientes que no cancelan, con métricas de precisión y recall altas y similares, aunque Random Forest tiene un recall ligeramente superior (0.90 vs 0.85), lo que reduce los falsos positivos en esta clase.
Falsos Positivos y Falsos Negativos:

Random Forest tiene menos falsos positivos (102 vs 151), lo que significa que comete menos errores al predecir incorrectamente que un cliente va a cancelar. KNN tiene ligeramente menos falsos negativos (180 vs 187), lo que significa que comete ligeramente menos errores al no identificar a clientes que sí cancelan.
Conclusión:

Considerando que el objetivo principal en un problema de predicción de cancelación de clientes suele ser identificar correctamente a los clientes propensos a cancelar para aplicar medidas de retención, el modelo Random Forest parece ser ligeramente mejor en este escenario.

Aunque el modelo KNN identifica a un número marginalmente mayor de verdaderos positivos (clientes que cancelan correctamente), el modelo Random Forest ofrece una mayor precisión en sus predicciones de cancelación (reduce los falsos positivos en la clase 'Yes') y una mejor exactitud general.

La elección final entre los modelos dependerá de la prioridad del negocio: si es más costoso un falso positivo (gastar recursos en retener a alguien que no se iba a ir) o un falso negativo (perder a un cliente que se iba a ir y no fue identificado).

En muchos casos de cancelación, reducir los falsos positivos para optimizar las campañas de retención es crucial, lo que inclinaría la balanza a favor de Random Forest en este caso, a pesar de su ligero menor recall en la clase de cancelación.

Se podrían explorar técnicas adicionales para mejorar el recall de la clase minoritaria en Random Forest, como ajustar los pesos de las clases o probar otros algoritmos.
## 8.ANÁLISIS VARIABLES RELEVANTES PREDICCIÓN DE CANCELACIÓN (CHURN)
Basándonos en la exploración inicial de los datos, el análisis de correlación y las visualizaciones dirigidas, podemos identificar y discutir las variables que parecen tener una influencia más significativa en la probabilidad de que un cliente cancele su servicio. Es crucial entender no solo qué variables están relacionadas con la cancelación, sino también cómo se relacionan y por qué podrían ser importantes desde una perspectiva de negocio.

Variables Clave y su Relación con la Cancelación (Churn):

TENENCIA

Hallazgo:

El análisis de correlación mostró una fuerte correlación negativa (-0.354) entre la antigüedad (tenencia) y la cancelación.
La visualización del boxplot de cuota_total por cancelación también respaldó esta idea indirectamente, ya que los clientes que no cancelan tienden a tener una cuota_total más alta, lo cual está fuertemente relacionado con una mayor antigüedad.
Análisis Crítico:

Esta es una de las variables más intuitivas y consistentemente importantes en la predicción de cancelación.
Los clientes que llevan más tiempo con la empresa generalmente han establecido una relación más sólida, están más acostumbrados al servicio y es menos probable que busquen alternativas de inmediato.
Han superado las posibles frustraciones iniciales y han encontrado valor a largo plazo.
Una baja tenencia, por el contrario, indica que el cliente aún se encuentra en una fase temprana de su relación con la empresa, donde la percepción del valor y la satisfacción pueden ser más volátiles.
Es más fácil para un cliente nuevo irse si no cumple sus expectativas iniciales.
Por lo tanto, la tenencia es un predictor fundamental de la lealtad del cliente.
Tipo de Contrato:

Hallazgo:

-El análisis de correlación mostró correlaciones negativas significativas con los contratos a largo plazo: contrato_Two year (-0.302) y contrato_One year (-0.178). -La visualización del countplot para contrato_One year demostró claramente que los clientes con contratos de un año (y por extensión, los de dos años) ** tienen una tasa de **cancelación mucho menor en comparación con aquellos sin este tipo de contrato (principalmente contratos mes a mes).

Análisis Crítico:

El tipo de contrato es un indicador directo del compromiso del cliente con la empresa.
Un contrato a largo plazo implica una obligación contractual que disuade la cancelación prematura.
Los clientes que eligen contratos de uno o dos años probablemente ya perciben un valor significativo en el servicio o están buscando estabilidad en sus costos.
Los contratos mes a mes, por otro lado, ofrecen flexibilidad total, lo que facilita a los clientes cambiar de proveedor si encuentran una oferta mejor, experimentan problemas con el servicio o simplemente ya no necesitan el servicio.
La fuerza de la correlación negativa con los contratos de dos años es particularmente importante, subrayando que un compromiso a más largo plazo es un fuerte indicador de retención.
Tipo de Internet (Fibra Óptica vs. Otros/Sin Internet):

Hallazgo:

La variable tipo_internet_Fiber optic mostró una correlación positiva moderada (0.307) con la cancelación, mientras que tipo_internet_No tuvo una correlación negativa (-0.228).
Análisis Crítico:

Este hallazgo es interesante y puede reflejar varios factores. Los servicios de fibra óptica suelen ser más rápidos y, a menudo, más caros.
Los clientes que optan por fibra pueden tener mayores expectativas de rendimiento y pueden ser más sensibles a problemas de servicio o a los costos si perciben que el valor no justifica el precio.
El mercado de fibra óptica puede ser más competitivo, con otros proveedores ofreciendo alternativas atractivas.
Por otro lado, los clientes sin servicio de internet (tipo_internet_No) tienen una correlación negativa con la cancelación, lo que sugiere que aquellos que solo tienen servicios básicos (como teléfono) son menos propensos a irse.
Esto podría deberse a que tienen menos opciones de proveedores para solo servicios telefónicos o sus necesidades son más simples y menos propensas a la insatisfacción.
La correlación de la fibra óptica resalta un segmento de clientes de alto valor pero potencialmente de alto riesgo.
Método de Pago (Cheque Electrónico):

Hallazgo:

La variable metodo_pago_Electronic check mostró una correlación positiva moderada (0.301) con la cancelación. -Otros métodos de pago como metodo_pago_Credit card (automatic) (-0.135) y metodo_pago_Mailed check (-0.091) tuvieron correlaciones negativas más débiles.

Análisis Crítico:

La fuerte correlación positiva con el cheque electrónico es un patrón común observado en muchos conjuntos de datos de cancelación.
Los clientes que utilizan cheques electrónicos tienden a ser más propensos a cancelar.
Esto podría deberse a que este método de pago es a menudo elegido por clientes con contratos mes a mes (que ya son más propensos a cancelar) o puede indicar un segmento demográfico o de comportamiento particular.
También podría ser que los problemas técnicos o la falta de automatización completa asociados con los cheques electrónicos generen fricción que contribuye a la insatisfacción.
Los métodos de pago automáticos (tarjeta de crédito, transferencia bancaria) muestran correlaciones negativas, lo que sugiere que la automatización del pago puede estar asociada con una mayor retención, quizás por conveniencia o porque los clientes más comprometidos optan por la automatización.
Cuota Mensual y Facturación Electrónica:

Hallazgo:

La cuota_mensual (0.193) y facturacion_electronica (0.191) mostraron correlaciones positivas más débiles pero aún relevantes con la cancelación.
Análisis Crítico:

Si bien no son tan fuertes como la tenencia o el tipo de contrato, estas variables sugieren que cuotas mensuales más altos y la preferencia por la facturación electrónica están asociados con una mayor probabilidad de cancelación.
Clientes con facturas mensuales elevadas pueden ser más sensibles a los precios y más propensos a buscar alternativas más económicas.
La facturación electrónica, aunque conveniente para muchos, podría estar asociada con clientes que son más activos en línea y, por lo tanto, más expuestos a ofertas de la competencia o más propensos a gestionar sus servicios digitalmente, incluida la cancelación.
Servicios Adicionales (Seguridad Online, Respaldo Online, Soporte Técnico, Streaming TV/Películas):

Hallazgo:

Varias variables indicando la ausencia de estos servicios (por ejemplo, seguridad_online_No internet service, streaming_tv_No internet service) mostraron correlaciones negativas significativas (alrededor de -0.228), lo cual está fuertemente relacionado con la ausencia de servicio de internet en general.
Las variables que indican la presencia de estos servicios (por ejemplo, seguridad_online_Yes (-0.171), soporte_tecnico_Yes (-0.165), respaldo_online_Yes (-0.082), proteccion_dispositivo_Yes (-0.066)) mostraron correlaciones negativas más débiles.
streaming_tv_Yes (0.063) y streaming_peliculas_Yes (0.061) tuvieron correlaciones positivas débiles.
Análisis Crítico:

Los servicios adicionales tienen un impacto variado.
La correlación negativa de la ausencia de estos servicios (cuando no hay servicio de internet) refuerza la idea de que los clientes con servicios básicos son menos propensos a cancelar.
La correlación negativa de la presencia de servicios como seguridad, respaldo y soporte técnico sugiere que estos servicios pueden actuar como "factores de adherencia" que aumentan la lealtad del cliente.
Los clientes que utilizan estos servicios pueden percibirlos como valiosos y una razón para quedarse.
Curiosamente, los servicios de streaming_tv y películas tienen correlaciones positivas débiles, lo que podría indicar que, si bien son populares, no necesariamente garantizan la retención y podrían estar asociados con clientes que buscan entretenimiento y pueden cambiar fácilmente si encuentran mejores opciones en otro lugar.
Variables con Correlación Débil (Género, Tiene Teléfono):

Hallazgo:

Variables como genero_Male (-0.008) y tiene_telefono (0.012) mostraron correlaciones muy cercanas a cero.
Análisis Crítico:

Estas variables parecen tener poca o ninguna relación lineal directa con la cancelación.
Si bien podrían tener interacciones complejas con otras variables, individualmente no son fuertes predictores de cancelación según este análisis de correlación.
## 9.CONCLUSIÓN GENERAL DEL ANÁLISIS DE VARIABLES

El análisis de correlación y las visualizaciones confirman que la predicción de la cancelación de clientes es un problema complejo influenciado por una combinación de factores. Las variables más relevantes y con mayor poder predictivo parecen estar relacionadas con:

El compromiso a largo plazo:** tenencia** (antigüedad) y tipo de contrato (especialmente contratos de dos años) son los predictores más fuertes de la retención.

Tipo de servicios utilizados:

-El tipo de internet (fibra óptica como riesgo, no internet como bajo riesgo) y la presencia de servicios de valor añadido como seguridad, respaldo y soporte técnico influyen en la probabilidad de cancelación.

Comportamiento de pago y costos:

El método de pago (cheque electrónico como riesgo, métodos automáticos como retención) y las cuotas mensuales más altas están asociados con una mayor propensión a cancelar.
La modelación predictiva (como la realizada con KNN y Random Forest) utiliza estas relaciones para construir un modelo que pueda estimar la probabilidad de cancelación para un cliente dado.
Los modelos como Random Forest, que pueden capturar relaciones no lineales e interacciones entre variables, a menudo funcionan bien en este tipo de problemas.
La dificultad para predecir la clase minoritaria (cancelación) en ambos modelos subraya la necesidad de técnicas adicionales si el objetivo principal es identificar a todos los clientes en riesgo (mejorar el recall de la clase 'Yes'), como el sobremuestreo de la clase minoritaria, el submuestreo de la clase mayoritaria, o el uso de algoritmos más adecuados para datos desbalanceados.
En resumen, un modelo efectivo de predicción de cancelación debe dar un peso significativo a la antigüedad del cliente y al tipo de contrato, al tiempo que considera el tipo de internet, el método de pago y los servicios adicionales como indicadores importantes del comportamiento y riesgo de cancelación del cliente.
## 10.PLANES ESTRATÉGICOS PARA REDUCIR CANCELACIÓN (CHURN)

##10.1.OBJETIVOS ESTRATÉGICOS GENERALES

###**Metas Cuantificables**
1. **Reducir tasa de cancelación** en segmentos de alto riesgo identificados
2. **Mejorar percepción de valor** en segmentos de alto gasto mensual
3. **Aumentar adopción** de contratos a largo plazo y métodos de pago automáticos
4. **Incrementar engagement** y uso de servicios de valor añadido

###**Segmentos Prioritarios (Según Datos del Análisis)**
- **CRÍTICO**: Contrato mes a mes + Cheque electrónico = **53.73% de cancelación**
- **ALTO RIESGO**: Fibra óptica sin servicios adicionales = **55.42% de cancelación**
- **OBJETIVO**: Contrato 2 años + Cheque postal = **0.80% de cancelación** (modelo a replicar)

---

##**IMPLEMENTACIÓN Y SEGUIMIENTO**

###**Fases de Implementación**
1. **Fase 1 (0-3 meses)**: Implementar detección temprana y onboarding mejorado
2. **Fase 2 (3-6 meses)**: Lanzar campañas de activación y bundling de servicios
3. **Fase 3 (6-12 meses)**: Ejecutar segmentación completa y programas VIP
4. **Fase 4 (12+ meses)**: Optimización continua y revisión estratégica de precios

### **Monitoreo Continuo**
- **Dashboards en tiempo real** de métricas de churn por segmento
- **Alertas automáticas** para clientes en riesgo
- **Reportes mensuales** de efectividad de cada estrategia
- **Análisis trimestral** de ROI y ajustes estratégicos

---

*Este plan integral aborda los factores de riesgo más críticos identificados en el análisis, priorizando acciones específicas y medibles para los segmentos de mayor vulnerabilidad a la cancelación.*

##10.2. PLANES ESTRATÉGICOS PARA REDUCIR CANCELACIÓN (CHURN)

##10.2. PLANES ESTRATÉGICOS PARA REDUCIR CANCELACIÓN (CHURN)

### **10.2.1.ESTRATEGIA 1: RETENCIÓN TEMPRANA Y MITIGACIÓN DE RIESGOS INICIALES**

**Objetivo Principal**

Dirigirse a **nuevos clientes con tenencia < 9 meses**, especialmente aquellos con características de alto riesgo:
- Contratos mes a mes
- Método de pago cheque electrónico
- Fibra óptica sin servicios adicionales
- Adultos mayores

###**Acciones Específicas**

####**Programa de Onboarding Mejorado**
- **Tutoriales personalizados** por segmento (especial atención a adultos mayores y servicios complejos como fibra óptica)
- **Soporte técnico proactivo** en las primeras semanas
- **Línea de contacto dedicada** para nuevas activaciones

####**Detección Temprana de Insatisfacción**
- **Encuestas de satisfacción programadas** a los 15 días, 3 meses y 6 meses
- **Sistema de alertas** para identificar problemas antes de que escalen a cancelaciones

####**Incentivos para Compromiso a Largo Plazo**
- **Descuentos significativos** para clientes mes a mes que cambien a contratos anuales
- **Servicios premium gratuitos** por tiempo limitado como incentivo de conversión

###**KPIs Clave**
- Reducción de tasa de cancelación en clientes < 9 meses
- % de nuevos clientes que migran a contratos a largo plazo
- Puntuaciones de satisfacción temprana

---

###10.2.2**ESTRATEGIA 2: ACTIVACIÓN DE SERVICIOS DE VALOR Y ENGAGEMENT CONTINUO**

###**Objetivo Principal**
Maximizar el engagement de clientes con:
- **Bajo engagement** (< 1.2 cuentas diarias)
- **Servicios de internet sin complementos** (sin Seguridad Online, Protección de Dispositivo o Soporte Técnico)

### **Acciones Específicas**

####**Campañas de Activación Digital**
- **Educación sobre beneficios** de servicios que reducen churn
- **Canales múltiples**: email, SMS, tutoriales web/app, webinars
- **Contenido adaptado** por segmento demográfico

####**Ofertas de Paquetes Estratégicos**
- **Bundles atractivos**: Internet + Seguridad + Protección a precio preferencial
- **Promociones cruzadas** basadas en el perfil de uso del cliente

####**Evaluación de Valor Percibido**
- **Análisis profundo** de por qué clientes de alto gasto (Fibra Óptica) no perciben valor suficiente
- **Ajustes en comunicación** de propuesta de valor

### **KPIs Clave**
- Aumento % de adopción de servicios complementarios
- Mejora en puntuación de engagement
- Reducción de churn en segmentos de bajo engagement previo

---


### 10.2.3.**ESTRATEGIA 3: SEGMENTACIÓN POR RIESGO Y VALOR PARA RETENCIÓN DIRIGIDA

###**Matriz de Segmentación**

####**ALTO RIESGO + ALTO VALOR** 
*Ejemplo: Fibra Óptica + Alto gasto + Contrato mes a mes + Cheque Electrónico*
- **Programa VIP de retención proactiva**
- **Gestor de cuentas dedicado**
- **Ofertas personalizadas** para migrar a contratos largos y pago automático
- **Resolución prioritaria** de problemas

####**ALTO RIESGO + VALOR MODERADO/BAJO**
*Ejemplo: Contrato mes a mes + Cheque Electrónico*
- **Campañas masivas segmentadas**
- **Descuentos temporales**
- **Beneficios por cambio** de método de pago
- **Simplificación de procesos**

####**BAJO RIESGO + ALTO VALOR**
*Ejemplo: Contrato 2 años + Pago Automático + Fibra + Servicios Adicionales*
- **Programas de fidelización**
- **Ofertas exclusivas por lealtad**
- **Acceso anticipado** a nuevos servicios
- **Comunicación VIP personalizada**

####**BAJO RIESGO + BAJO VALOR**
*Ejemplo: Sin internet + Bajo gasto*
- **Ofertas de upgrade** a planes básicos DSL
- **Énfasis en facilidad de uso** y bajo costo
- **Mantenimiento** de experiencia positiva actual

---

###10.2.4**ESTRATEGIA 4: REVISIÓN ESTRATÉGICA DE PRECIOS Y PROPUESTA DE VALOR

###**Análisis y Optimización**

####**Revisión de Estructura de Precios**
- **Análisis más allá** del umbral de $35 mensual
- **Enfoque en segmentos** de alto riesgo independiente de cuota actual
- **Evaluación precio-valor percibido** especialmente en Fibra Óptica

####**Creación de Planes de Entrada Estratégicos**
- **Ofertas para nuevos clientes** que fomenten contratos largos desde el inicio
- **Incentivos automáticos** para métodos de pago preferenciales

####**Bundling de Servicios de Valor**
- **Paquetes integrados** con servicios de seguridad y soporte
- **Precios atractivos** que reduzcan sensibilidad al costo individual
- **Propuesta de valor sólida** y comunicada claramente

####**Comunicación del Valor Total**
- **Enfoque holístico**: velocidad, seguridad, soporte, fiabilidad
- **Justificación clara** especialmente para servicios premium como Fibra Óptica
- **Educación continua** sobre beneficios recibidos

---

## TECNOLOGIAS USADAS

• Python 3 

• Google Colab 

• Cuaderno Jupiter 

• VS Code

• Pandas 

• Matplotlib 

• NumPy

• Seaborn

• pandas
 
• sklearn

• Qwen3

• Claude sonnet4

• Chatgpt

##LIBRERIAS USADAS
- pandas
- numpy 
- matplotlib.pyplot 
- seaborn 
- import requests
- sklearn.model_selection import train_test_split
- sklearn.preprocessing import StandardScaler
- sklearn.linear_model import LogisticRegression
- sklearn.ensemble import 
- RandomForestClassifier
- sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


## INSTALACION
Clonar este repositorio en tu máquina local: git clone https://github.com/giacomo1960/Desafio2_TelecomX
Instalar my-project con npm

  npm install my-project
  cd my-project
