"""
Bienestar Unisabana — Entrenamiento del clasificador de riesgo
==============================================================
Genera un dataset sintético basado en el quiz de 8 preguntas,
entrena un RandomForestClassifier y lo exporta como función
JavaScript pura (sin dependencias) usando m2cgen.

Uso:
    cd prototipo
    pip install scikit-learn m2cgen numpy
    python train_model.py

Salida:
    model.js  → incluir en index.html antes de app.js

Reemplazo por datos reales:
    Cuando el sistema tenga respuestas recopiladas de estudiantes,
    reemplaza `X` e `y` (sección "DATOS") por el CSV real y
    vuelve a ejecutar este script. El resto del código no cambia.
"""

import numpy as np
import itertools
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import m2cgen as m2c


# ══════════════════════════════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════════════════════════════
# Actualmente: dataset sintético con todos los perfiles posibles del quiz.
# Cada pregunta toma valores 1 (bajo riesgo), 2 (moderado) o 3 (alto).
# 3^8 = 6 561 combinaciones posibles.
#
# Para usar datos reales cuando los tengas:
#   df = pd.read_csv("respuestas_estudiantes.csv")
#   X  = df[["q1","q2","q3","q4","q5","q6","q7","q8"]].values
#   y  = df["nivel_riesgo"].values   # "verde", "amarillo" o "rojo"
# ══════════════════════════════════════════════════════════════════════════════

X = np.array(list(itertools.product([1, 2, 3], repeat=8)))   # (6561, 8)


def etiquetar(row: np.ndarray) -> str:
    """
    Lógica de dominio validada con Bienestar Unisabana.
    Espejo de la función clasificar() original en app.js.
    """
    total = int(row.sum())
    q8    = int(row[7])   # pregunta sobre abandono — indicador crítico
    if q8 == 3:
        return "rojo"
    if total <= 11:
        return "verde"
    if total <= 18:
        return "amarillo"
    return "rojo"


y = np.array([etiquetar(r) for r in X])

# Ruido realista (5 %)
# Simula perfiles ambiguos en la frontera entre niveles.
# Fuerza al RF a aprender fronteras probabilísticas en lugar de memorizar la regla.
rng = np.random.default_rng(42)
clases = ["verde", "amarillo", "rojo"]
noise_idx = np.where(rng.random(len(y)) < 0.05)[0]
for i in noise_idx:
    vecinos = [c for c in clases if c != y[i]]
    y[i] = rng.choice(vecinos)

print(f"Dataset: {X.shape[0]} muestras  |  distribución: { {c: (y==c).sum() for c in clases} }")


# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=25,       # 25 árboles: balance tamaño JS / capacidad
    max_depth=5,           # profundidad acotada para mantener model.js legible
    min_samples_leaf=3,
    class_weight="balanced",  # compensa el desbalance verde<<amarillo/rojo
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# ── Validación cruzada ────────────────────────────────────────────────────────
cv_scores = cross_val_score(rf, X, y, cv=StratifiedKFold(5), scoring="accuracy")
print(f"\nValidación cruzada (5-fold):  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\nReporte en conjunto de prueba (20 %):")
print(classification_report(y_test, rf.predict(X_test), digits=3))

# Orden de clases (alfabético): amarillo=0, rojo=1, verde=2
print("Orden de clases (rf.classes_):", list(rf.classes_))


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAR A JAVASCRIPT con m2cgen
# ══════════════════════════════════════════════════════════════════════════════
# m2cgen convierte el RF en una función JS pura, sin librerías externas.
# Para un clasificador, score(input) devuelve un array de votos acumulados
# (uno por clase). El argmax identifica la clase predicha.

js_body = m2c.export_to_javascript(rf)

class_order = list(rf.classes_)   # ["amarillo", "rojo", "verde"]
class_order_js = json.dumps(class_order)

# Métricas a incrustar como comentario en el archivo generado
acc_test = (rf.predict(X_test) == y_test).mean()
meta_comment = (
    f"  Accuracy (test 20%): {acc_test:.3f}  |  "
    f"CV 5-fold: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  |  "
    f"n_estimators={rf.n_estimators}, max_depth={rf.max_depth}"
)

output_js = f"""/* =============================================
   MODELO RANDOM FOREST — GENERADO AUTOMÁTICAMENTE
   No editar manualmente. Ejecutar train_model.py para regenerar.
   {meta_comment}
   =============================================

   Clases en orden de índice: {class_order_js}
   score(input) → array de votos [amarillo, rojo, verde]
   El índice con más votos es la clase predicha.
   ============================================= */

/* Etiquetas en el mismo orden que rf.classes_ */
const RF_CLASSES = {class_order_js};

/* ── Función generada por m2cgen (no editar) ── */
{js_body}

/* ── Wrapper: recibe array de 8 respuestas (1-3) y retorna "verde"/"amarillo"/"rojo" ── */
function clasificarRF(respuestas) {{
  const votos = score(respuestas);                          // array de votos por clase
  const idxMax = votos.indexOf(Math.max(...votos));         // clase con más votos
  return RF_CLASSES[idxMax];
}}
"""

out_path = Path(__file__).parent / "model.js"
out_path.write_text(output_js, encoding="utf-8")
print(f"\nModelo exportado a: {out_path}")
print("  Incluye model.js en index.html antes de app.js para usarlo.")
