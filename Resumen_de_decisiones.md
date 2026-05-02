# Resumen de Decisiones – Challenge Data Scientist Entel

## Decisiones más relevantes

### 1. Separación temporal train/test
Se respetó la temporalidad: train con junio y julio 2014, test con agosto 2014. Esto simula producción real y evita *data leakage*. El modelo aprende del pasado para predecir el futuro.

### 2. Segmentación ARPU por percentiles (P25 / P75 / P90)
Se eligió segmentación por percentiles sobre K-means porque es interpretable para el negocio, estable ante nuevos datos y no requiere reentrenamiento. Se generaron 4 segmentos: Bajo, Medio, Alto y Premium.

### 3. Imputación de nulos con 0
Las variables de uso (minutos, datos, recargas) tienen nulos que se interpretan como "no usó ese mes". Imputar con 0 es semánticamente correcto en contexto TELCO.

### 4. Comparación de 5 modelos con métricas de discriminación ampliadas
Se compararon Regresión Logística, Random Forest, LightGBM, LightGBM+SMOTETomek y LightGBM (Optuna). Las métricas de evaluación son AUC-ROC, Average Precision, KS Statistic y F1-Churn (threshold óptimo), además de la ganancia de negocio. El KS Statistic es especialmente relevante en scoring de churn: indica en qué score se produce la mayor separación entre churners y no-churners.

### 5. Manejo del desbalance: SMOTETomek como alternativa a `scale_pos_weight`
Se evaluó SMOTETomek (SMOTE + Tomek Links) frente al uso de `scale_pos_weight`. SMOTETomek crea muestras sintéticas en la región de decisión y elimina pares ambiguos, forzando una frontera más nítida. El modelo sobre datos rebalanceados se entrena sin `scale_pos_weight` para evitar doble penalización.

### 6. Optimización de hiperparámetros con Optuna orientada a ganancia
Se usó Optuna (50 trials, TPE Sampler) para optimizar LightGBM directamente por ganancia de negocio con umbral EV, en lugar de AUC-ROC. Optimizar AUC puede mejorar el ranking global sin mejorar la rentabilidad real; optimizar ganancia alinea el modelo con el objetivo de negocio.

### 7. Estrategia de contacto: umbral de valor esperado (EV) en lugar de Top-2000 fijo
En lugar de contactar siempre a 2.000 clientes, solo se contacta si `P(churn) > 33.3%`. Ese umbral es el break-even exacto por contacto: `$1.000 / (10% × $30.000) = 0.333`. Contactar clientes con P menor garantiza pérdida por ese contacto. El número final de contactos es variable (≤ 2.000), ajustado a la calidad del score del modelo.

### 8. Métrica de negocio como criterio de evaluación y selección del modelo final
Se definió una función de ganancia neta con umbral EV:  
`Ganancia = TP_ev × 10% × $30.000 − N_ev × $1.000`  
donde `N_ev` = clientes contactados con `P(churn) > 33.3%`. Esta es la métrica principal de éxito y el criterio de selección del modelo final. El break-even requiere al menos 667 churners reales en 2.000 contactos (`$2.000.000 / $3.000 = 667`).

---

## Supuestos clave

- Los nulos en variables de uso representan "sin consumo", no errores de captura.
- La tasa de éxito del 10% es uniforme para todos los churners contactados (no varía por segmento).
- El valor recuperado de $30.000 CLP es fijo por cliente (no proporcional al ARPU).
- No hay clientes duplicados entre meses (cada fila es un cliente en un mes dado).

---

## Principales riesgos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Data drift en producción | Alto | Monitorear PSI mensualmente |
| Nulos por error de captura (no por ausencia de uso) | Medio | Validar con equipo de datos |
| Modelo sobreajustado a jul/ago 2014 | Medio | Validación cruzada temporal |
| Tasa de retención del 10% varía por segmento | Alto | Personalizar campaña por segmento ARPU |
| Sesgo por clientes inactivos pre-churn | Bajo | Se incluyó `inactivo_total` como feature |
