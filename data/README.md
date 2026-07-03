# Dataset: Elliptic Bitcoin

Este proyecto usa el **Elliptic Data Set**, un grafo de transacciones de Bitcoin
con ~203k nodos (transacciones) y ~234k aristas (flujos de pago), etiquetado
como lícito / ilícito / desconocido por Elliptic en colaboración con el MIT.

## Cómo obtenerlo

1. Crea una cuenta en Kaggle si no tienes una.
2. Ve a: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
3. Descarga el ZIP y descomprímelo en `data/raw/`. Deberías tener:
   - `elliptic_txs_features.csv`   (features de cada transacción, sin nombres de columna)
   - `elliptic_txs_classes.txt`    (etiqueta por transacción: 1=ilícito, 2=lícito, unknown)
   - `elliptic_txs_edgelist.csv`   (aristas del grafo: txId1 -> txId2)

   Alternativa vía CLI (si tienes `kaggle` configurado con tu API token):
   ```bash
   kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
   ```

## Notas importantes sobre el dataset

- Las 166 features por transacción están anonimizadas por Elliptic. Las
  primeras 94 son "locales" (info de la propia transacción), las 72
  restantes son "agregadas" (estadísticas del vecindario a 1 salto). Esto
  es relevante para el feature engineering de la Semana 3: no vas a poder
  interpretar cada columna individualmente, así que la explicabilidad vía
  SHAP se apoya en importancia relativa, no en significado de negocio de
  cada feature.
- El dataset está organizado en 49 "time steps" (~2 semanas cada uno). Esto
  te da la posibilidad de hacer un split temporal real (train en steps
  1-34, val en 35-42, test en 43-49) en vez de un split aleatorio — evita
  el data leakage temporal que mencioné en el roadmap.
- Solo ~21% de las transacciones tienen etiqueta conocida. De esas, la
  clase ilícita es ~2% del total etiquetado — de ahí el desbalance de
  clases que justifica usar AUC-PR en vez de accuracy.

Una vez tengas los 3 archivos en `data/raw/`, corre:
```bash
python src/graph/build_graph.py
```
