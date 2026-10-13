# Pipeline experimental – Detección de fraude con tarjeta de crédito

Tesis de Maestría en Data Science & Business Analytics (UMECIT). Comparación de 4 modelos x 3 estrategias de
balanceo en dos datasets abiertos de Kaggle (ULB e IEEE-CIS). Última actualización: 13/10/2026.

## Estado del proyecto
| Elemento | Estado |
|---|---|
| EDA (`01_eda.py`) | Hecho (figuras en `resultados/eda_*.png`) |
| Experimento ULB (`02_modelado.py`) | Hecho: 3 semillas (42, 7, 2024), 36 combinaciones |
| Experimento IEEE-CIS (`03_ieee_cis.py`) | Hecho: 3 semillas (42, 7, 2024), 36 combinaciones, partición temporal |
| Bootstrap e intervalos de confianza (`05_bootstrap.py`) | Hecho: B = 1000, 3 semillas por dataset |
| Figuras y tabla final (`04_figuras.py`) | Hechas con las 72 predicciones (las de IEEE-CIS no se publican; ver más abajo) |
| Sensibilidad del costo (`06_sensibilidad_costo.py`) | Hecha (costo administrativo y comercial) |
| Caracterización de los datos (`07_caracterizacion.py`) | Hecha (`resultados/caracterizacion_*`, `car_*.png`) |
| Análisis de errores (`08_analisis_errores.py`) | Hecho: calibración, captura en las alertas, fraudes por monto y umbral por costo (exploratorio) |
| Figuras conceptuales (`09_figuras_conceptuales.py`) | Hechas (`fig_*.png`) |
| Contraste SMOTE antes de dividir (`10_smote_antes_de_dividir.py`) | Hecho: 3 semillas en ULB (`smote_antes_de_dividir.csv`) |
| Grilla ampliada en IEEE-CIS (`11_grilla_ampliada_ieee.py`) | Hecha: semilla 42 (`ieee_cis_grillaamp_*.csv`) |
| Referencias | `referencias_verificadas.md` |


## Estructura
| Archivo | Función |
|---|---|
| `common.py` | Carga de ULB, partición, métricas, umbral por F1 y costo (`COSTO_FP`) |
| `experimento.py` | Motor común: 4 modelos x 3 balanceos, grillas, umbral, guardado de predicciones y reanudación |
| `01_eda.py` | Análisis exploratorio de ULB |
| `02_modelado.py` | Experimento en ULB (partición aleatoria estratificada) |
| `03_ieee_cis.py` | Experimento en IEEE-CIS (partición temporal) |
| `04_figuras.py` | Tabla final y figuras comparativas |
| `05_bootstrap.py` | IC del 95 % del AUC-PR y contrastes pareados para H1 y H2 |
| `06_sensibilidad_costo.py` | Sensibilidad del costo al costo de falso positivo (Ca) y al costo comercial (λ) |
| `07_caracterizacion.py` | Caracterización de los conjuntos de datos (estructura, calidad, desbalance, montos, estabilidad temporal) |
| `08_analisis_errores.py` | Calibración, captura en las alertas de mayor puntuación, fraudes no detectados por monto, curvas PR y umbral por costo (exploratorio) |
| `09_figuras_conceptuales.py` | Figuras conceptuales (flujo, decisión del detector, SMOTE); no usan los datos |
| `10_smote_antes_de_dividir.py` | Contraste: SMOTE aplicado a todo ULB antes de dividir (protocolo con fuga de información) |
| `11_grilla_ampliada_ieee.py` | Sensibilidad de la grilla de hiperparámetros en IEEE-CIS (una semilla) |

## Datos (`data/`, no se publican en el repositorio)
- `ulb/creditcard.csv`: Kaggle mlg-ulb/creditcardfraud (284 807 filas, 492 fraudes, 1 081 duplicados exactos).
  Licencia indicada en Kaggle: DbCL v1.0 (contenidos) y ODbL (base). La moneda de `Amount` no está documentada.
- `ieee_cis/train_transaction.csv` y `train_identity.csv`: Kaggle ieee-fraud-detection (590 540 filas, 3,5 % de fraude).
  Sujeto a las reglas de la competencia; licencia no verificada. `TransactionAmt` en USD (confirmar en Kaggle).
- Los datos hay que descargarlos desde Kaggle; no se redistribuyen (ver `data/README.md` y `NOTICE.md`).

## Ejecución
```
pip install scikit-learn imbalanced-learn xgboost matplotlib seaborn pandas python-docx
python 01_eda.py
python 02_modelado.py 42 7 2024       # ULB
python 03_ieee_cis.py 42 7 2024       # IEEE-CIS (sin argumentos usa solo 42 y 7)
python 05_bootstrap.py 1000           # requiere las predicciones de ULB e IEEE-CIS (las de IEEE-CIS se generan localmente)
python 04_figuras.py
python 06_sensibilidad_costo.py       # usa las predicciones disponibles localmente
python 07_caracterizacion.py
python 08_analisis_errores.py       # requiere resultados/predicciones/
python 09_figuras_conceptuales.py
python 10_smote_antes_de_dividir.py 42 7 2024   # contraste (ULB)
python 11_grilla_ampliada_ieee.py 42  # sensibilidad de la grilla (IEEE-CIS; varias horas)
```
Versiones usadas: Python 3.12.10, scikit-learn 1.9.1, imbalanced-learn 0.14.2, XGBoost 3.4.1, pandas 3.0.5,
numpy 2.5.2, matplotlib 3.11.2, seaborn 0.13.2. No ejecutar los scripts con `python -I` (impide importar `common.py`).

**Predicciones publicadas.** `resultados/predicciones/` incluye solo las de ULB (`ulb__*.npz`; etiquetas reales, montos y
probabilidades del conjunto de prueba), bajo los términos de `NOTICE.md`. Las de IEEE-CIS **no se publican**: las reglas de la
competencia prohíben redistribuir los datos, y esos archivos contienen etiquetas reales y montos de transacciones. Para
reproducir los resultados de IEEE-CIS hay que descargar los datos de Kaggle, aceptar sus reglas y ejecutar
`python 03_ieee_cis.py 42 7 2024` (varias horas); `05_bootstrap.py` y `06_sensibilidad_costo.py` leerán entonces las
predicciones que se generen localmente. Los CSV de `resultados/` son agregados (métricas, intervalos y costos) y no contienen
datos por transacción. Tampoco se publican las predicciones de la grilla ampliada de IEEE-CIS (`ieee_cis_grillaamp__*.npz`), que se generan localmente.

**Reanudación:** si el proceso se corta (ocurrió en una VDI), relanzar el mismo comando; las combinaciones con archivo
en `resultados/predicciones/` se saltan.

**Advertencia sobre los CSV de IEEE-CIS:** `03_ieee_cis.py` reescribe `ieee_cis_resultados_crudos.csv` y
`ieee_cis_resumen_modelos.csv` solo con las semillas de esa ejecución. Los resultados finales se obtuvieron en dos
ejecuciones (semillas 42 y 7; luego 2024) y los CSV se fusionaron a mano . Para reproducir de una vez, ejecutar con las tres semillas (`python 03_ieee_cis.py 42 7 2024`). Las
predicciones (`.npz`) son por semilla y no se pisan; el bootstrap y la sensibilidad leen de ahí.

## Diseño experimental
- Modelos: regresión logística, árbol de decisión, Random Forest, XGBoost.
- Balanceo (nunca combinado): sin balanceo, SMOTE (k=5; en IEEE-CIS parcial, minoritaria al 20 % de la mayoritaria),
  pesos de clase. Solo sobre entrenamiento.
- Partición 70/15/15: aleatoria estratificada (ULB) y cronológica (IEEE-CIS). Escalado, imputación y codificación
  ajustados solo con entrenamiento. En IEEE-CIS la partición es fija; la semilla solo cambia la aleatoriedad de los modelos.
- Hiperparámetros: grilla pequeña evaluada en validación (AUC-PR). Umbral: máx. F1 en validación. Métricas en test.
- Métrica principal AUC-PR; complementarias AUC-ROC, precisión, exhaustividad, F1 y costo.
- Costo = monto de los fraudes no detectados + costo de cada falso positivo. Base: `COSTO_FP` = 5 unidades monetarias
  por falso positivo (supuesto; Bahnsen et al., 2013, usaron 2,50 EUR y probaron de 1 a 5). `06_sensibilidad_costo.py`
  usa costo de FP = Ca + λ·monto, con Ca en {1; 2,5; 5; 10} y λ en {0; 0,25; 1}. Los montos no se convierten de moneda.
  Si faltan predicciones de un dataset, el script calcula los datasets disponibles y omite el faltante con un aviso.

## Resultados (AUC-PR medio entre semillas; detalle en `resultados/` y en el Capítulo IV)
- ULB: XGBoost 0,84–0,86; Random Forest ≈ 0,84; regresión logística 0,73–0,76; árbol 0,64–0,77.
- IEEE-CIS (temporal): XGBoost 0,49–0,50; Random Forest 0,42–0,48; árbol 0,32–0,38; regresión logística ≈ 0,14.
- H1 (ensambles > referencias): se cumple en ambos datasets; con más fuerza en IEEE-CIS (36 de 36 comparaciones con IC > 0)
  que en ULB (34 de 36).
- H2 (SMOTE > sin balanceo): no se sostiene de forma general; solo Random Forest mejora en IEEE-CIS.
- Contraste de protocolo: aplicar SMOTE a todo ULB antes de dividir lleva el AUC-PR a casi 1,0 (0,997 a 1,000 en la prueba completa) y lo infla
  incluso en las filas reales de la prueba (Random Forest 0,976 y XGBoost 0,985, frente a 0,84 con el protocolo correcto): es fuga de información.
- Grilla ampliada en IEEE-CIS (semilla 42): el AUC-PR cambia en promedio 0,01 (máximo 0,05); no cambian la mejor combinación ni el orden de los modelos.
- Costo: el ranking es robusto a Ca entre 1 y 10; con costo comercial alto (λ = 1) en IEEE-CIS la mejor combinación no
  ahorra frente a no usar modelo, porque el umbral se eligió por F1.

## Desviaciones respecto a la Entrega Parcial II (declararlas en la tesis)
- Tuning por validación, no por validación cruzada (SMOTE + CV exigiría pipeline por fold).
- Se añade la estrategia "pesos de clase", la partición temporal, el segundo dataset y el análisis de costo.
- IEEE-CIS: SMOTE parcial, grillas reducidas y Random Forest más ligero (100 árboles, submuestreo del 30 %), por costo de cómputo.
- No se guardaron predicciones de validación; por eso el umbral no se reoptimiza en la sensibilidad del costo.

## Uso de inteligencia artificial
Este código se escribió, depuró y ejecutó con apoyo de Claude (Anthropic) bajo la dirección del autor, quien definió las
decisiones metodológicas. La declaración y la bitácora de uso de IA están en el apéndice de la tesis (UMECIT).

## Correcciones de citas detectadas
Adejoh (2024) → Khalid, Owoh, Uthmani, Ashawa, Osamor y Adejoh (2024); Kim (2024) → Feng y Kim (2024).
