# Referencias verificadas - Tesis: detección de fraude con tarjeta de crédito (ML supervisado)

Fecha de verificación: 2026-10-06. Metadatos bibliográficos contrastados contra Crossref (api.crossref.org), Semantic Scholar, portales institucionales, dblp/ACM, arXiv y páginas de editorial. Notas importantes:
- Varias páginas de MDPI, IEEE Xplore, ScienceDirect y Kaggle bloquearon la descarga directa (HTTP 403/418/JS). Por eso, en algunas fichas el "resultado cuantitativo" se indica como "no extraído" en lugar de inventarlo. Antes de citar cifras en la tesis, abrir el artículo y copiar el número exacto.
- "Scopus: confirmado" solo cuando un portal institucional muestra conteo de citas Scopus para el artículo. En los demás casos: "por confirmar" (verificar en scopus.com/sources).
- Los preprints (arXiv) se marcan como tales; no son revistas indexadas.

---

## A) ANTECEDENTES (2021-2026)

### A1. Ileberi, Sun y Wang (2021)
- Referencia: Ileberi, E., Sun, Y., & Wang, Z. (2021). Performance Evaluation of Machine Learning Methods for Credit Card Fraud Detection Using SMOTE and AdaBoost. IEEE Access, 9, 165286-165294. DOI: 10.1109/ACCESS.2021.3134330 (https://doi.org/10.1109/access.2021.3134330)
- Dataset: tarjetahabientes europeos (tipo ULB) más un dataset sintético de fraude muy desbalanceado (tamaños no extraídos).
- Objetivo: evaluar si AdaBoost combinado con SMOTE mejora la detección de fraude de varios clasificadores.
- Metodología: SMOTE + SVM, LR, RF, XGBoost, DT y Extra Trees, cada uno con AdaBoost; métricas: exactitud, recall, precisión, MCC y AUC.
- Resultado: los modelos potenciados con AdaBoost superaron a los métodos previos (cifra exacta no extraída; consultar el artículo).
- Conclusión/limitación: el boosting mejora el desempeño con datos balanceados vía SMOTE; no usa partición temporal ni AUC-PR como métrica principal.
- Aporte a la tesis: antecedente directo de SMOTE con LR, DT, RF y XGBoost en el dataset ULB.
- Scopus: confirmado (portal pure.uj.ac.za muestra 236 citas Scopus). Fuente: https://pure.uj.ac.za/en/publications/performance-evaluation-of-machine-learning-methods-for-credit-car/

### A2. Khalid, Owoh, Uthmani, Ashawa, Osamor y Adejoh (2024)
- Referencia: Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J., & Adejoh, J. (2024). Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach. Big Data and Cognitive Computing, 8(1), 6. DOI: 10.3390/bdcc8010006 (https://www.mdpi.com/2504-2289/8/1/6)
- Dataset: transacciones de tarjetahabientes europeos (ULB).
- Objetivo: construir un ensamble robusto para detección de fraude con datos desbalanceados.
- Metodología: ensamble de SVM, KNN, RF, Bagging y Boosting; submuestreo y SMOTE.
- Resultado: según resumen de buscador, RF y Boosting alcanzaron 99,98 % de exactitud en prueba con SMOTE (cifra de snippet; confirmar en el PDF). Exactitud no es adecuada con desbalance extremo.
- Conclusión/limitación: el ensamble supera a los modelos individuales; los autores mencionan concept drift y falsos positivos/negativos como retos pendientes; evalúa con exactitud, precisión, recall y F1, no AUC-PR.
- Aporte: ejemplo para contrastar por qué la tesis usa AUC-PR en lugar de exactitud; compara submuestreo vs SMOTE.
- Scopus: confirmado (portal researchonline.gcu.ac.uk muestra 218 citas Scopus). Fuentes: https://researchonline.gcu.ac.uk/en/publications/enhancing-credit-card-fraud-detection-an-ensemble-machine-learnin/ ; https://westminsterresearch.westminster.ac.uk/item/w80vz/enhancing-credit-card-fraud-detection-an-ensemble-machine-learning-approach

### A3. Feng y Kim (2024)
- Referencia: Feng, X., & Kim, S.-K. (2024). Novel Machine Learning Based Credit Card Fraud Detection Systems. Mathematics, 12(12), 1869. DOI: 10.3390/math12121869 (https://www.mdpi.com/2227-7390/12/12/1869)
- Dataset: tarjetahabientes europeos (ULB).
- Objetivo: reducir tamaño y complejidad del conjunto de entrenamiento sin perder exactitud.
- Metodología: "Compact Data Learning" (CDL) para reducción de características frente a otros algoritmos ML y métodos de reducción, con manejo de desbalance.
- Resultado: CDL supera a varios algoritmos y métodos de reducción (cifra no extraída).
- Conclusión/limitación: aporte teórico y práctico; sin evaluación temporal.
- Aporte: referencia de comparación de estrategias de preprocesamiento/selección de variables sobre ULB.
- Scopus: confirmado (portal research.mpu.edu.mo muestra 37 citas Scopus). Fuente: https://research.mpu.edu.mo/en/publications/novel-machine-learning-based-credit-card-fraud-detection-systems/

### A4. Mienye y Sun (2023)
- Referencia: Mienye, I. D., & Sun, Y. (2023). A Deep Learning Ensemble With Data Resampling for Credit Card Fraud Detection. IEEE Access, 11, 30628-30638. DOI: 10.1109/ACCESS.2023.3262020
- Dataset: por confirmar (el resumen no lo nombra).
- Objetivo: mitigar el desbalance con un ensamble profundo y remuestreo híbrido.
- Metodología: stacking de LSTM y GRU con MLP como metaaprendiz; balanceo con SMOTE-ENN.
- Resultado: sensibilidad 1,000 y especificidad 0,997.
- Conclusión/limitación: supera a clasificadores ML usuales; sensibilidad de 1,0 sugiere posible optimismo/fuga si el remuestreo se aplicó antes de la partición (verificar en el artículo).
- Aporte: ejemplo de SMOTE-ENN como alternativa a SMOTE y advertencia metodológica sobre fuga de datos.
- Scopus: por confirmar. Fuente de metadatos: Crossref y Semantic Scholar (api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ACCESS.2023.3262020).

### A5. Albalawi y Dardouri (2025)
- Referencia: Albalawi, T., & Dardouri, S. (2025). Enhancing credit card fraud detection using traditional and deep learning models with class imbalance mitigation. Frontiers in Artificial Intelligence, 8. DOI: 10.3389/frai.2025.1643292 (https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1643292/full)
- Dataset: Kaggle Credit Card Fraud Detection (284.807 transacciones, 0,17 % fraude) y PaySim.
- Objetivo: comparar modelos tradicionales y de aprendizaje profundo con mitigación del desbalance.
- Metodología: LR, DT, RF, XGBoost y red profunda con focal loss.
- Resultado: RF mejor en Kaggle con exactitud 99,95 %, F1 0,8256 y ROC-AUC 0,9759; XGBoost precisión 91,67 % y recall 95 %; la red profunda tuvo solo 4 falsos positivos.
- Conclusión/limitación: validada también en PaySim; no describe partición temporal; el modelo profundo necesita GPU para tiempo real (~25 ms).
- Aporte: es casi el mismo conjunto de modelos que la tesis (LR, DT, RF, XGBoost) sobre ULB; sirve como línea base comparativa.
- Scopus: por confirmar.

### A6. Zuberi et al. (2026)
- Referencia: Zuberi, N., Rahman, M. M., Rashid, S. U., Ahsan, M. N., Rahman, M. S., & Mollah, M. H. O. R. (2026). A robust machine learning framework for detecting temporal drift in financial fraud prevention. Scientific Reports, 16, 27776. DOI: 10.1038/s41598-026-58285-5 (https://www.nature.com/articles/s41598-026-58285-5 ; texto completo en https://pmc.ncbi.nlm.nih.gov/articles/PMC13542085/)
- Dataset: ULB (284.807 transacciones, 0,173 % fraude).
- Objetivo: evaluar modelos y estrategias de remuestreo bajo partición cronológica y cuantificar el drift temporal.
- Metodología: 85 combinaciones modelo-muestreador (12 clasificadores clásicos, MLP, FraudNet, 7 ensambles; SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, SMOTEENN, SMOTETomek); partición cronológica 70/15/15; PSI, KS, SHAP; umbral optimizado en validación.
- Resultado: MLP sin remuestreo F2 0,7722; Soft Voting con SMOTE MCC 0,8060 y AUC-ROC 0,9703; LightGBM pasa de F2 0,0745 a 0,7588 con SMOTEENN; PSI train-test 0,088; "ningún método de sobremuestreo domina".
- Conclusión/limitación: el dataset cubre solo 48 horas, por lo que el drift medido es una cota inferior.
- Aporte: respalda comparar SMOTE vs pesos de clase/sin remuestreo, usar partición cronológica y reconocer la limitación temporal de ULB.
- Scopus: por confirmar.

### A7. Ileberi y Sun (2025)
- Referencia: Ileberi, E., & Sun, Y. (2025). Performance Analysis of Cost-Sensitive Oversampling Mechanisms for Credit Card Fraud Detection. IEEE Access, 13, 202655-202664. DOI: 10.1109/ACCESS.2025.3637132
- Dataset: "datos reales de tarjetas de crédito" (nombre exacto por confirmar).
- Objetivo: incorporar el costo de la transacción en el sobremuestreo.
- Metodología: cinco técnicas de sobremuestreo (SMOTE, variantes, ADASYN) en versión estándar y sensible al costo, con RF, XGBoost, LightGBM, red neuronal y regresión logística.
- Resultado: el sobremuestreo sensible al costo "aumenta sustancialmente el ahorro total" y reduce el costo medio de clasificación errónea (cifras no extraídas); mayores ganancias en modelos neuronales y lineales.
- Conclusión/limitación: da recomendaciones para métodos conscientes del costo; la partición temporal no fue verificada.
- Aporte: base para el análisis de costo de negocio y para comparar SMOTE con variantes sensibles al costo.
- Scopus: por confirmar. Fuente: Semantic Scholar (api.semanticscholar.org/graph/v1/paper/DOI:10.1109/ACCESS.2025.3637132).

### A8. Fan y Boonen (2026)
- Referencia: Fan, X.-Y., & Boonen, T. J. (2026). Machine Learning Algorithms for Credit Card Fraud Detection: Cost-Sensitive Evaluation and Model Selection. Asia-Pacific Journal of Risk and Insurance. DOI: 10.1515/apjri-2025-0025 (volumen/páginas no disponibles en Crossref al momento)
- Dataset: por confirmar.
- Objetivo: seleccionar modelos con evaluación Fβ sensible al costo en lugar de F1.
- Metodología: varios clasificadores con SMOTE/Borderline SMOTE, ensamble soft-voting de LR+RF, evaluación Top-K e interpretabilidad.
- Resultado: soft-voting LR+RF con SMOTE logró Fβ = 0,8287; XGBoost con Borderline SMOTE bueno individualmente.
- Conclusión/limitación: la intervención óptima depende del modelo.
- Aporte: justifica evaluación orientada a costo y Top-K operativo en la tesis.
- Scopus: por confirmar.

### A9. Sang (2026)
- Referencia: Sang, V. N. T. (2026). Enhancing Credit Card Fraud Detection under Severe Class Imbalance using Cost-Sensitive Learning and Threshold Optimization. EAI Endorsed Transactions on Intelligent Systems and Machine Learning Applications, 3. DOI: 10.4108/eetismla.12078 (https://publications.eai.eu/index.php/ismla/article/view/12078)
- Dataset: Kaggle 2013 (ULB, muy desbalanceado) y Kaggle 2023 (más balanceado).
- Objetivo: mejorar la detección sin remuestreo sintético.
- Metodología: XGBoost sensible al costo y red neuronal con BCE ponderada y Focal Loss; separación entre optimización del ranking y selección del umbral.
- Resultado: PR-AUC de 0,7809 a 0,8815 y falsos positivos de 21 a 13 sobre un conjunto de prueba con 98 fraudes.
- Conclusión/limitación: separar ranking y umbral es útil; los patrones de fraude cambian con el tiempo.
- Aporte: ejemplo directo de pesos de clase vs SMOTE con AUC-PR como métrica principal.
- Scopus: por confirmar.

### A10. Kang y Kim (2026)
- Referencia: Kang, J., & Kim, K. (2026). Deployment-Oriented Interpretable Fraud Detection via Hybrid Explainable Boosting Machines with Concept-Raw Fusion on the IEEE-CIS Benchmark. Applied Sciences, 16, 5809. DOI: 10.3390/app16125809 (metadatos: Crossref y Semantic Scholar)
- Dataset: IEEE-CIS Fraud Detection.
- Objetivo: lograr un detector interpretable apto para producción.
- Metodología: EBM híbridos con banco de 63 conceptos causales y modelos "estudiante" guiados por maestro; validación cronológica fuera de tiempo y holdout disjunto por pseudo-entidad.
- Resultado: PR-AUC 0,407 ± 0,003 (híbrido) vs 0,383 ± 0,002 (solo variables crudas); inferencia 9,7 veces más rápida que XGBoost; 154 a 71 variables.
- Conclusión/limitación: compromiso interpretable, no un sistema de monitoreo de producción.
- Aporte: referencia de PR-AUC realista (~0,4) en IEEE-CIS con validación temporal; ancla de expectativas.
- Scopus: por confirmar (MDPI Applied Sciences).

### A11. Han y Wu (2026) - PREPRINT
- Referencia: Han, X., & Wu, C. (2026). Validation-Stage Combinatorial Fusion Analysis for Imbalanced Credit-Card Fraud Detection. arXiv:2606.10393 (https://arxiv.org/abs/2606.10393). Preprint, no revisado por pares.
- Dataset: IEEE-CIS, partición 60/20/20 sin fuga.
- Objetivo: evaluar si el Combinatorial Fusion Analysis mejora a los modelos de árboles potenciados.
- Metodología: fusión ponderada por diversidad de RF, XGBoost y LightGBM; comparación con stacking y soft voting.
- Resultado: AUC-ROC 0,9405, AUPRC 0,6699, F1 0,6373.
- Conclusión/limitación: es más útil para elegir un subconjunto complementario; preprint. La partición es 60/20/20 (el artículo indica "leakage-free"; no se verificó si es cronológica).
- Aporte: referencia de AUPRC en IEEE-CIS con RF y XGBoost.
- Scopus: no aplica (preprint).

### A12. Dastidar, Caelen y Granitzer (2024)
- Referencia: Dastidar, K. G., Caelen, O., & Granitzer, M. (2024). Machine Learning Methods for Credit Card Fraud Detection: A Survey. IEEE Access, 12, 158939-158965. DOI: 10.1109/ACCESS.2024.3487298
- Dataset: revisión (propone además un marco de generación de datos con GAN).
- Objetivo: organizar la literatura según desbalance, patrones de comportamiento cambiantes y representación del contexto.
- Metodología: revisión sistemática con taxonomía.
- Resultado: no aplica (survey).
- Conclusión/limitación: señala la escasez de datasets públicos de calidad y la necesidad de manejar la evolución del comportamiento.
- Aporte: soporte teórico para concept drift, desbalance y limitaciones de los datasets públicos (coautor Caelen es del grupo ULB/Worldline).
- Scopus: por confirmar.

### A13. El Hajjami y Diallo (2025)
- Referencia: El Hajjami, S., & Diallo, G. (2025). SMOTE-OSBNR: An Effective Approach for Imbalanced Credit Card Fraud Detection. IEEE Access, 13, 183503-183518. DOI: 10.1109/ACCESS.2025.3624961
- Dataset: siete datasets de fraude (nombres no extraídos).
- Objetivo: reducir el ruido generado por SMOTE.
- Metodología: SMOTE seguido de OSBNR (filtrado de instancias de la clase mayoritaria ruidosas o solapadas) con Gradient Boosted Trees, XGBoost y RF.
- Resultado: AUC-ROC de 0,999 y F1 de 0,990 (mejor caso).
- Conclusión/limitación: AUC-ROC cercano a 1 sugiere evaluación optimista (posible remuestreo antes de partir); no usa AUC-PR.
- Aporte: ilustra las críticas a SMOTE (ruido/solape) que motivan compararlo con pesos de clase.
- Scopus: por confirmar.

### A14. Diaconu y Pele (2026)
- Referencia: Diaconu, D., & Pele, D. T. (2026). Explainable AI in Financial Fraud Detection: Evidence from the IEEE-CIS Fraud Dataset. Proceedings of the International Conference on Business Excellence, 20, 118-132. DOI: 10.2478/picbe-2026-0015
- Dataset: IEEE-CIS.
- Objetivo: comparar explicabilidad entre modelos tabulares y basados en grafos.
- Metodología: XGBoost, CatBoost y GraphSAGE con SHAP y ablación de nodos/aristas.
- Resultado: XGBoost y CatBoost lideran en exactitud predictiva (sin cifra extraída).
- Conclusión/limitación: se recomienda combinar modelos tabulares y relacionales.
- Aporte: uso de IEEE-CIS con XGBoost y explicabilidad SHAP.
- Scopus: por confirmar (conferencia).

### A15. Almazroi y Ayub (2023)
- Referencia: Almazroi, A. A., & Ayub, N. (2023). Online Payment Fraud Detection Model Using Machine Learning Techniques. IEEE Access, 11, 137188-137203. DOI: 10.1109/ACCESS.2023.3339226
- Dataset: tres datasets de transacciones financieras (nombres no extraídos).
- Objetivo: detectar fraude en pagos en línea con un modelo profundo optimizado.
- Metodología: ResNeXt + GRU (RXT), autoencoder + ResNet (EARN), SMOTE y optimización Jaya.
- Resultado: supera a métodos existentes en 10-18 % entre métricas.
- Conclusión/limitación: eficiencia computacional mantenida; la comparación no se verificó con particiones temporales.
- Aporte: uso de SMOTE en varios datasets de pagos.
- Scopus: por confirmar.

### A16. Wu (2026) - PREPRINT
- Referencia: Wu, C. (2026). Class Weighting versus Amount Conditioning in Credit-Card Fraud Detection: A Dollar-Metric Study with a Temporal Explanation Audit. arXiv:2607.14686 (https://arxiv.org/abs/2607.14686). Preprint.
- Dataset: dos datasets cronológicos (nombres no extraídos).
- Objetivo: decidir si el monto debe ponderar el entrenamiento o reordenar alertas.
- Metodología: XGBoost con distintas estrategias de ponderación; auditoría SHAP mes a mes.
- Resultado: el class weighting estándar rinde igual que el condicionamiento por monto; reordenar alertas por score x monto da la mayor recuperación en dólares.
- Conclusión/limitación: el monto sirve como variable y para ordenar alertas, no como regla de ponderación; mayor volatilidad de atribución SHAP en fraudes.
- Aporte: apoya el análisis de costo de negocio en dólares y el uso de pesos de clase con XGBoost.
- Scopus: no aplica (preprint).

Cobertura de los criterios solicitados: IEEE-CIS (A10, A11, A14); evaluación temporal/drift (A6, A10, A12, A16); aprendizaje sensible al costo (A7, A8, A9, A16); SMOTE vs otras estrategias (A2, A6, A7, A13). Publicaciones en revista 2021-2026: 13; preprints: 2 (A11, A16); conferencia: 1 (A14).

---

## B) VERIFICACIÓN DE REFERENCIAS METODOLÓGICAS

| # | Referencia dada | Estado | Datos correctos / fuente |
|---|---|---|---|
| 1 | Saito & Rehmsmeier (2015) PLOS ONE | CONFIRMADA | Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432. DOI: 10.1371/journal.pone.0118432 (Crossref) |
| 2 | Chen & Guestrin (2016) XGBoost KDD | CONFIRMADA | Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proc. 22nd ACM SIGKDD (KDD '16), pp. 785-794. DOI: 10.1145/2939672.2939785 (Crossref); arXiv:1603.02754 |
| 3 | Breiman (2001) Random Forests | CONFIRMADA | Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. DOI: 10.1023/A:1010933404324 (Crossref) |
| 4 | Dal Pozzolo et al. (2014) | CONFIRMADA | Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, 41(10), 4915-4928. DOI: 10.1016/j.eswa.2014.02.026 (Crossref) |
| 5 | Fernández et al. (2018) SMOTE 15-year, JAIR | CORREGIDA (título completo, volumen y páginas) | Fernández, A., García, S., Herrera, F., & Chawla, N. V. (2018). SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary. Journal of Artificial Intelligence Research, 61, 863-905. DOI: 10.1613/jair.1.11192 (Crossref; jair.org) |
| 6 | Elkan (2001) foundations of cost-sensitive learning, IJCAI | CONFIRMADA | Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. IJCAI'01: Proc. 17th International Joint Conference on Artificial Intelligence, pp. 973-978. ACM DL: 10.5555/1642194.1642224 (https://dl.acm.org/doi/10.5555/1642194.1642224). No tiene DOI Crossref estándar; verificado vía búsqueda ACM DL/dblp (conf/ijcai/Elkan01). |
| 7 | Davis & Goadrich (2006) ICML | CONFIRMADA | Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. Proc. 23rd ICML (ICML '06), pp. 233-240. DOI: 10.1145/1143844.1143874 (Crossref) |
| 8 | Adejoh (2024) BDCC 8(1) 6 | CORREGIDA (autoría) | La revista, volumen, número y artículo son correctos, pero Adejoh es el ÚLTIMO autor. Cita correcta: Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J., & Adejoh, J. (2024). Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach. Big Data and Cognitive Computing, 8(1), 6. DOI: 10.3390/bdcc8010006 |
| 9 | Kim (2024) Mathematics 12(12) 1869 | CORREGIDA (autoría) | El primer autor es Feng, no Kim. Cita correcta: Feng, X., & Kim, S.-K. (2024). Novel Machine Learning Based Credit Card Fraud Detection Systems. Mathematics, 12(12), 1869. DOI: 10.3390/math12121869 (Crossref) |
| 10 | Chawla et al. (2002) JAIR 16 | CONFIRMADA | Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357. DOI: 10.1613/jair.953 (Crossref) |

---

## C) LICENCIA / TÉRMINOS DE USO DE LOS DATASETS

1. Dataset ULB "Credit Card Fraud Detection" (Kaggle, mlg-ulb/creditcardfraud)
   - Licencia indicada: Database Contents License (DbCL) v1.0; la base está bajo Open Database License (ODbL) y los contenidos bajo DbCL. Fuente (resultado de búsqueda con extracto de la página de Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (la descarga directa de la página no mostró la licencia por ser contenido dinámico; la identificación DbCL v1.0 proviene del extracto del buscador).
   - Texto de la DbCL v1.0 (https://opendatacommons.org/licenses/dbcl/1-0/): concede licencia mundial, gratuita, no exclusiva, perpetua e irrevocable para cualquier acto restringido por derechos de autor sobre los contenidos, con uso comercial permitido y sin obligación de atribución propia; la ODbL de la base puede imponer obligaciones adicionales (atribución y share-alike para la base de datos).
   - Procedencia: recopilado en colaboración entre Worldline y el Machine Learning Group de ULB (septiembre 2013, 284.807 transacciones, 492 fraudes). Recomendación: citar a Dal Pozzolo et al. (2015, entre otros) como pide la página de Kaggle (cita exacta por verificar en la página).
   - Conclusión: el uso académico está permitido.

2. Dataset de la competencia IEEE-CIS Fraud Detection (Kaggle)
   - NO PUDE VERIFICAR el texto de las reglas. Las páginas https://www.kaggle.com/competitions/ieee-fraud-detection (y /rules, /data) cargan por JavaScript y la descarga devolvió solo el título. No hay una licencia explícita verificada en fuente primaria.
   - Datos verificables: es una "Research Prediction Competition" organizada por IEEE Computational Intelligence Society con datos de Vesta Corporation (fuente: resultado de búsqueda de la página de la competencia). Una réplica en IEEE DataPort (doi:10.21227/y5e7-wp63) no declara licencia y requiere suscripción (https://ieee-dataport.org/documents/ieee-cis-fraud-detection).
   - Un preprint (preprints.org) menciona CC BY y otras publicaciones Apache 2.0; son afirmaciones de terceros no verificadas y no deben citarse como licencia.
   - Acción recomendada: iniciar sesión en Kaggle, aceptar las reglas, abrir la pestaña Rules (sección "Data Access and Use") y copiar literalmente la cláusula; solo si menciona "Academic Research" afirmar el permiso académico. Muchos artículos revisados por pares (A10, A11, A14) usan este dataset con fines de investigación, pero eso no sustituye a la licencia.

---

## RESUMEN (10 líneas)
1. Verificadas 10/10 referencias metodológicas (7 confirmadas, 3 con correcciones: Fernández, Adejoh/Khalid, Kim/Feng).
2. Corrección clave: Adejoh (2024) es el último autor; el primero es Khalid. Kim (2024) es en realidad Feng y Kim.
3. Fernández (2018): título completo "SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary", JAIR 61, 863-905.
4. Elkan (2001) confirmado por ACM DL/dblp (pp. 973-978), sin DOI Crossref.
5. Antecedentes: 16 fichas (13 revistas, 1 conferencia, 2 preprints arXiv), todas con metadatos contrastados.
6. Scopus confirmado solo en 3 (A1, A2, A3) vía portales institucionales; el resto "por confirmar".
7. Varias cifras (A1, A3, A7, A14) NO se extrajeron porque MDPI/IEEE/ScienceDirect bloquearon la lectura; hay que copiarlas del PDF antes de citar. La cifra de A2 (99,98 %) viene de un extracto de buscador.
8. Cobertura IEEE-CIS: A10, A11, A14; temporal/drift: A6, A10, A12, A16; costo: A7, A8, A9, A16.
9. Licencia ULB: DbCL v1.0 (uso académico y comercial permitido). Licencia IEEE-CIS: NO verificada (Kaggle carga por JS); se debe leer la pestaña Rules manualmente.
10. No se incluyó ninguna referencia que no apareciera en Crossref, Semantic Scholar, ACM, arXiv o página institucional/editorial.

## ADICIÓN 08/10/2026: costo de falso positivo
- Bahnsen, A. C., Stojanovic, A., Aouada, D., & Ottersten, B. (2013). Cost sensitive credit card fraud detection using Bayes minimum risk. 2013 12th International Conference on Machine Learning and Applications, 333-338. IEEE. https://doi.org/10.1109/ICMLA.2013.68. VERIFICADA en Crossref (DOI, autores, año, páginas, editorial). Texto completo leído: FP = costo administrativo Ca (analizar la transacción y contactar al titular); Ca = 2,50 EUR en el análisis principal y sensibilidad de 1,00 a 5,00 EUR (tabla VI); datos de un procesador europeo; el texto revisado no justifica el origen del 2,50. Falta confirmar si está indexada en Scopus.
- Moneda: IEEE-CIS TransactionAmt en USD (según fuentes secundarias; confirmar en la página de datos de Kaggle). ULB: la descripción de Kaggle solo dice "Amount"; la moneda NO está documentada (no afirmar euros).
- Fuente secundaria para el costo comercial: The Paypers, citado en Ravelin (https://www.ravelin.com/blog/reduce-false-positives-fraud): "25 % de los compradores rechazados por error se pasan a un competidor". Verificar en la fuente original antes de citar.
- Wallny (2022), HICSS, "False Positives in Credit Card Fraud Detection: Measurement and Mitigation": solo se leyó el resumen; verificar autoría completa y DOI antes de citar.
