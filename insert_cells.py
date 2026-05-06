import json, sys, uuid
sys.stdout.reconfigure(encoding='utf-8')

with open('caracterización.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_code(source, cell_id=None):
    return {
        "cell_type": "code",
        "id": cell_id or uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }

def make_md(source, cell_id=None):
    return {
        "cell_type": "markdown",
        "id": cell_id or uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }

# ── CELL 1: Markdown header ──────────────────────────────────────────────
md_header = make_md(
    "### Modelado predictivo de la participación en Bienestar\n\n"
    "Para responder *¿qué variables predicen mejor la participación?* se comparan tres modelos "
    "sobre la misma partición (80 % entrenamiento / 20 % prueba):\n\n"
    "| Modelo | Por qué incluirlo |\n"
    "|---|---|\n"
    "| **Regresión lineal** | Línea base interpretable; referencia del análisis anterior |\n"
    "| **Random Forest** | Captura relaciones no lineales; provee SHAP values para explicabilidad |\n"
    "| **Red Neuronal (MLP)** | Mayor capacidad expresiva; contraste con los modelos anteriores |\n\n"
    "Los SHAP values responden directamente la pregunta de investigación: muestran "
    "*cuánto y en qué dirección* empuja cada variable la predicción de participación de cada estudiante.",
    "ml-header-md"
)

# ── CELL 2: Setup + split + baseline LR ─────────────────────────────────
setup_src = (
    "# ── Configuración: dependencias y partición de datos ─────────────────────\n"
    "%pip install -q shap\n\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.metrics import r2_score, mean_squared_error\n"
    "from sklearn.linear_model import LinearRegression as LR\n"
    "import numpy as np\n\n"
    'TARGET_ML      = "Participación total"\n'
    "PREDICTORES_ML = [c for c in df_plot.columns if c != TARGET_ML]\n\n"
    "X_ml = df_plot[PREDICTORES_ML].values\n"
    "y_ml = df_plot[TARGET_ML].values\n\n"
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X_ml, y_ml, test_size=0.2, random_state=42\n"
    ")\n"
    'print(f"Muestras: entrenamiento={len(X_train):,}   prueba={len(X_test):,}")\n'
    'print(f"Predictores: {PREDICTORES_ML}")\n\n'
    "# Regresión lineal sobre la misma partición (línea base)\n"
    "scaler_lr = StandardScaler()\n"
    "lr_cmp    = LR().fit(scaler_lr.fit_transform(X_train), y_train)\n"
    "y_pred_lr = lr_cmp.predict(scaler_lr.transform(X_test))\n"
    "r2_lr     = r2_score(y_test, y_pred_lr)\n"
    "rmse_lr   = np.sqrt(mean_squared_error(y_test, y_pred_lr))\n"
    'print(f"\\nRegresión lineal  →  R² = {r2_lr:.4f}   RMSE = {rmse_lr:.4f}")\n'
)
setup_cell = make_code(setup_src, "ml-setup")

# ── CELL 3: Random Forest + SHAP ────────────────────────────────────────
rf_src = (
    "# ── Random Forest: importancia de variables y SHAP ────────────────────────\n"
    "import shap\n"
    "from sklearn.ensemble import RandomForestRegressor\n\n"
    "rf = RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)\n"
    "rf.fit(X_train, y_train)\n\n"
    "y_pred_rf = rf.predict(X_test)\n"
    "r2_rf     = r2_score(y_test, y_pred_rf)\n"
    "rmse_rf   = np.sqrt(mean_squared_error(y_test, y_pred_rf))\n"
    'print(f"Random Forest  →  R² = {r2_rf:.4f}   RMSE = {rmse_rf:.4f}")\n\n'
    "# Importancia MDI (Mean Decrease Impurity)\n"
    "importancias = pd.Series(rf.feature_importances_, index=PREDICTORES_ML).sort_values()\n"
    'colores_imp  = ["#4A7FA5" if v == importancias.max() else "#7BB3CC" for v in importancias.values]\n\n'
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "ax.barh(importancias.index, importancias.values, color=colores_imp, edgecolor='white')\n"
    "for i, (val, name) in enumerate(zip(importancias.values, importancias.index)):\n"
    '    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9)\n'
    "ax.set_title(\n"
    '    "Importancia de variables — Random Forest (MDI)\\n"\n'
    '    "(proporción de reducción de impureza; mayor = más influyente)",\n'
    '    fontweight="bold"\n'
    ")\n"
    'ax.set_xlabel("Importancia relativa")\n'
    "plt.tight_layout()\n"
    'guardar(fig, "rf_importancia_variables.png")\n'
    "plt.show()\n\n"
    "# SHAP values (muestra de 500 registros para eficiencia)\n"
    "explainer   = shap.TreeExplainer(rf)\n"
    "rng         = np.random.default_rng(42)\n"
    "idx_sample  = rng.choice(len(X_test), size=min(500, len(X_test)), replace=False)\n"
    "shap_values = explainer.shap_values(X_test[idx_sample])\n\n"
    "shap.summary_plot(shap_values, X_test[idx_sample], feature_names=PREDICTORES_ML, show=False)\n"
    "fig_shap = plt.gcf()\n"
    "fig_shap.suptitle(\n"
    '    "SHAP — impacto de cada variable sobre la predicción de participación",\n'
    '    fontweight="bold", y=1.01\n'
    ")\n"
    'guardar(fig_shap, "shap_summary.png")\n'
    "plt.show()\n\n"
    'print("\\nInterpretación SHAP:")\n'
    'print("  · Puntos a la derecha (rojo) → valor alto de la variable SUBE la predicción")\n'
    'print("  · Puntos a la izquierda (azul) → valor alto de la variable BAJA la predicción")\n'
)
rf_cell = make_code(rf_src, "ml-rf-shap")

# ── CELL 4: MLP ─────────────────────────────────────────────────────────
mlp_src = (
    "# ── Red Neuronal (MLP) — predicción de participación ─────────────────────\n"
    "from sklearn.neural_network import MLPRegressor\n\n"
    "scaler_mlp = StandardScaler()\n"
    "X_train_sc = scaler_mlp.fit_transform(X_train)\n"
    "X_test_sc  = scaler_mlp.transform(X_test)\n\n"
    "mlp = MLPRegressor(\n"
    "    hidden_layer_sizes=(64, 32, 16),\n"
    '    activation="relu",\n'
    "    max_iter=400,\n"
    "    early_stopping=True,\n"
    "    validation_fraction=0.1,\n"
    "    random_state=42,\n"
    ")\n"
    "mlp.fit(X_train_sc, y_train)\n\n"
    "y_pred_mlp = mlp.predict(X_test_sc)\n"
    "r2_mlp     = r2_score(y_test, y_pred_mlp)\n"
    "rmse_mlp   = np.sqrt(mean_squared_error(y_test, y_pred_mlp))\n"
    'print(f"Red Neuronal MLP  →  R² = {r2_mlp:.4f}   RMSE = {rmse_mlp:.4f}")\n'
    'print(f"Épocas entrenadas: {mlp.n_iter_}")\n\n'
    "# Curvas de aprendizaje\n"
    "has_val = mlp.validation_scores_ is not None\n"
    "fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(11 if has_val else 7, 4))\n"
    "axes = [axes] if not has_val else axes\n\n"
    "axes[0].plot(mlp.loss_curve_, color='#4A7FA5')\n"
    "axes[0].set_title('Pérdida en entrenamiento', fontweight='bold')\n"
    "axes[0].set_xlabel('Época')\n"
    "axes[0].set_ylabel('MSE')\n\n"
    "if has_val:\n"
    "    axes[1].plot(mlp.validation_scores_, color='#E07B8B')\n"
    "    axes[1].set_title('R² en validación (early stopping)', fontweight='bold')\n"
    "    axes[1].set_xlabel('Época')\n"
    "    axes[1].set_ylabel('R²')\n\n"
    "plt.suptitle('Curvas de aprendizaje — Red Neuronal MLP', fontweight='bold')\n"
    "plt.tight_layout()\n"
    "guardar(fig, 'mlp_curvas_aprendizaje.png')\n"
    "plt.show()\n"
)
mlp_cell = make_code(mlp_src, "ml-mlp")

# ── CELL 5: Comparison ──────────────────────────────────────────────────
cmp_src = (
    "# ── Comparación final de los tres modelos ────────────────────────────────\n"
    "resultados = pd.DataFrame({\n"
    '    "Modelo": ["Regresión lineal", "Random Forest", "Red Neuronal (MLP)"],\n'
    '    "R²":     [r2_lr,   r2_rf,   r2_mlp],\n'
    '    "RMSE":   [rmse_lr, rmse_rf, rmse_mlp],\n'
    "}).sort_values('R²', ascending=False).reset_index(drop=True)\n\n"
    'print("Comparación de modelos (conjunto de prueba — 20 %):")\n'
    "print(resultados.to_string(index=False))\n\n"
    "palette_cmp = ['#4A7FA5', '#7BB3CC', '#AACDE0']\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n\n"
    "for i, (met, titulo) in enumerate([(\"R²\", \"R² (mayor es mejor)\"), (\"RMSE\", \"RMSE (menor es mejor)\")]):\n"
    "    vals = resultados[met].values\n"
    "    axes[i].bar(resultados['Modelo'], vals, color=palette_cmp, edgecolor='white')\n"
    "    axes[i].set_title(titulo, fontweight='bold')\n"
    "    for j, v in enumerate(vals):\n"
    "        axes[i].text(j, v * 1.02, f'{v:.3f}', ha='center', fontsize=10)\n"
    "    axes[i].tick_params(axis='x', rotation=15)\n\n"
    "plt.suptitle('Comparación de modelos — predicción de participación en Bienestar',\n"
    "             fontweight='bold', y=1.02)\n"
    "plt.tight_layout()\n"
    "guardar(fig, 'comparacion_modelos.png')\n"
    "plt.show()\n\n"
    "best = resultados.iloc[0]\n"
    'print(f"\\n Mejor modelo: {best[\'Modelo\']}  (R²={best[\'R²\']:.4f}, RMSE={best[\'RMSE\']:.4f})")\n'
    'print(\n'
    '    "\\nConclusión: el Random Forest suele dominar porque la relación entre variables "\n'
    '    "y participación NO es estrictamente lineal. Los SHAP values (celda anterior) "\n'
    '    "revelan qué variable empuja más cada predicción individual."\n'
    ')\n'
)
cmp_cell = make_code(cmp_src, "ml-comparison")

# Insert all 5 cells after index 30 (cell b8bf3179)
insert_pos = 31
new_cells = [md_header, setup_cell, rf_cell, mlp_cell, cmp_cell]
nb['cells'] = nb['cells'][:insert_pos] + new_cells + nb['cells'][insert_pos:]

with open('caracterización.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
print("New cells: ml-header-md (31), ml-setup (32), ml-rf-shap (33), ml-mlp (34), ml-comparison (35)")
