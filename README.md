# Data Scientist Challenge – Entel

## Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm imbalanced-learn optuna jupyter
```

## Estructura

```
Data_Scientist_Challenge_v1.5.1/
├── dataset_prueba.csv              # Dataset original (299.997 registros)
├── data_dict.csv                   # Diccionario de datos
├── decisiones.md                   # Resumen de decisiones de modelado
├── README.md
└── notebooks/
    ├── 01_EDA.ipynb                # Análisis exploratorio (Partes 1–5)
    ├── 02_Modelo_Predictivo.ipynb  # Modelamiento y priorización (Partes 1–10)
    └── lista_final_clientes.csv    # Output: clientes priorizados para campaña
```

## Ejecución

Desde el directorio raíz del challenge:

```bash
cd notebooks
jupyter notebook
```

Ejecutar en orden:
1. `01_EDA.ipynb` – EDA, segmentación ARPU, feature engineering, limpieza
2. `02_Modelo_Predictivo.ipynb` – Modelos, evaluación, optimización, priorización

El notebook 2 es **independiente**: incluye toda la preparación de datos internamente y puede ejecutarse sin depender del estado del notebook 1.

> **Nota:** La celda de Optuna (50 trials) tarda ~3–5 minutos en ejecutarse.

## Output principal

El modelo genera `notebooks/lista_final_clientes.csv` con los clientes priorizados para la campaña de retención de agosto 2014.

| Columna | Descripción |
|---|---|
| `rank` | Posición en el ranking (mayor P(churn) primero) |
| `mobile_number` | Identificador del cliente |
| `prob_churn` | Probabilidad de churn predicha por el modelo |
| `ev_neto_CLP` | Valor esperado neto por contacto = `P(churn) × 10% × $30.000 − $1.000` |
| `contactar` | 1 si `P(churn) > 33.3%` (EV positivo), 0 si no |
| `arpu` | ARPU del cliente en agosto 2014 |
| `churn_real` | Churn real (disponible en datos de test; para evaluación) |

## Modelos evaluados

| Modelo | Descripción |
|---|---|
| Logistic Regression | Baseline interpretable con `class_weight='balanced'` |
| Random Forest | Ensemble con `class_weight='balanced'` |
| LightGBM | Boosting con `scale_pos_weight` |
| LightGBM + SMOTETomek | Boosting sobre datos rebalanceados (SMOTE + Tomek Links) |
| LightGBM (Optuna) | Boosting con hiperparámetros optimizados por ganancia EV |

## Estrategia de contacto

En lugar de contactar siempre a 2.000 clientes, se usa un **umbral de valor esperado**:

```
Contactar solo si P(churn) > costo / (retention_rate × valor)
                           = $1.000 / (10% × $30.000) = 33.3%
```

Esto garantiza que cada contacto tenga valor esperado positivo. El número final de contactos es variable (≤ 2.000). Break-even: mínimo 667 churners reales en 2.000 contactos.

## Resultados clave

- **Modelo final:** seleccionado dinámicamente por máxima ganancia con umbral EV
- **Métricas reportadas:** AUC-ROC, Average Precision, KS Statistic, F1-Churn
- **Estrategia:** umbral EV `P(churn) > 33.3%` → N contactos variable
- **Ganancia neta esperada:** ver resumen ejecutivo al final del notebook 2
