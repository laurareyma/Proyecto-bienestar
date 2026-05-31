import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(
    page_title="Proyecto Bienestar | Unisabana",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #001C64;
    border-right: 3px solid #193F9E;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: #193F9E !important; }

/* Metric boxes */
.metric-box {
    background: linear-gradient(135deg, #001C64, #193F9E);
    border-radius: 12px;
    padding: 20px 12px 16px;
    text-align: center;
    color: #FFFFFF;
    margin-bottom: 8px;
    border: 1px solid #9DC4FF;
    box-shadow: 0 2px 8px rgba(0,28,100,0.18);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 6px;
}
.metric-box h2 {
    font-size: 1.65rem; font-weight: 700; margin: 0;
    color: #FFFFFF !important; white-space: nowrap; line-height: 1.2;
}
.metric-box p {
    font-size: 0.78rem; margin: 0; opacity: 0.88;
    line-height: 1.3; max-width: 140px; color: #FFFFFF !important;
}

/* Typography */
.section-title {
    font-size: 1.5rem; font-weight: 700; color: #001C64;
    margin-bottom: 4px; border-left: 4px solid #26A3DD; padding-left: 10px;
}
h1, h2, h3 { color: #001C64 !important; }

/* Divider */
.divider { border: none; border-top: 2px solid #9DC4FF; margin: 12px 0 20px; }

/* Links */
a { color: #26A3DD !important; }
a:hover { color: #70C3E9 !important; }

/* Buttons */
.stButton > button {
    background-color: #001C64; color: #FFFFFF;
    border: none; border-radius: 6px;
}
.stButton > button:hover { background-color: #193F9E; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "Data")
GRAPHICS = os.path.join(BASE, "src", "graphics")

# Paleta institucional Universidad de la Sabana
C_AZUL_SABANA    = "#001C64"
C_AZUL_CONTRASTE = "#193F9E"
C_AZUL_CLARO     = "#9DC4FF"
C_AZUL_CIELO     = "#E0EDFF"
C_ACENTO         = "#26A3DD"
C_ACENTO_HOVER   = "#70C3E9"
C_GRIS_OSCURO    = "#6D6D6D"
C_ERROR          = "#FF2114"

SABANA_SCALE       = [[0, C_AZUL_CIELO], [0.5, C_AZUL_CLARO], [1, C_AZUL_SABANA]]
SABANA_QUALITATIVE = [C_AZUL_CONTRASTE, C_ACENTO, C_AZUL_CLARO,
                      C_AZUL_SABANA, C_ACENTO_HOVER, C_GRIS_OSCURO]

PAM_COL  = "promedio_acumulado_programa_principal(pam)"
PCP_COL  = "promedio_semestral_programa_principal(pcp)"
CRED_COL = "creditos_inscritos"


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos…")
def load_all():
    asist_old = pd.read_csv(os.path.join(DATA, "asistencia_2015_2022_limpia.csv"))
    asist_new = pd.read_csv(os.path.join(DATA, "asistencia_2023_2025_limpia.csv"))
    caract    = pd.read_csv(os.path.join(DATA, "caracterizacion_limpia.csv"))
    perfiles  = pd.read_csv(os.path.join(DATA, "perfiles_estudiantiles.csv"))

    # Normalizar columnas del archivo 2023-2025 al esquema común
    col_map = {
        "ciclo_lectivo":              "periodo",
        "estrategia_para_acreditacion": "estrategia_programa_o_servicio",
        "tipo_de_actividad":          "actividad",
    }
    asist_new = asist_new.rename(columns={k: v for k, v in col_map.items() if k in asist_new.columns})

    if "ano" not in asist_new.columns:
        if "periodo" in asist_new.columns:
            asist_new["ano"] = pd.to_numeric(
                asist_new["periodo"].astype(str).str[:4], errors="coerce"
            ).astype("Int64")
        else:
            asist_new["ano"] = pd.NA

    common = ["ano", "periodo", "jefatura", "estrategia_programa_o_servicio", "actividad", "codigo"]
    asistencia = pd.concat(
        [asist_old[[c for c in common if c in asist_old.columns]],
         asist_new[[c for c in common if c in asist_new.columns]]],
        ignore_index=True,
    )
    return asistencia, caract, perfiles


@st.cache_data(show_spinner="Entrenando modelo predictivo…")
def train_predictor(_perfiles):
    try:
        from sklearn.linear_model import LinearRegression
        feat_cols = [c for c in [PAM_COL, PCP_COL, CRED_COL] if c in _perfiles.columns]
        target = "participacion_total"
        if target not in _perfiles.columns or not feat_cols:
            return None, feat_cols
        df = _perfiles[feat_cols + [target]].dropna()
        model = LinearRegression().fit(df[feat_cols].values, df[target].values)
        return model, feat_cols
    except Exception:
        return None, []


@st.cache_data(show_spinner="Entrenando regresión logística…")
def train_logistic_model(_caract, _asistencia):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        df_p1 = _caract.merge(
            _asistencia[["codigo", "actividad"]].dropna(),
            on="codigo", how="inner"
        )

        feat_cols = ["sexo", "ciudad_direccion_fisica", PAM_COL]
        target = "actividad"
        missing = [c for c in feat_cols + [target] if c not in df_p1.columns]
        if missing:
            return None, None, f"Columnas faltantes: {missing}"

        df_p1 = df_p1[feat_cols + [target]].dropna()
        if len(df_p1) < 50:
            return None, None, "Datos insuficientes tras filtrar NaN."

        X = df_p1[feat_cols]
        y = df_p1[target]

        cat_lr = ["sexo", "ciudad_direccion_fisica"]
        num_lr = [PAM_COL]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_lr),
                ("num", "passthrough", num_lr),
            ]
        )
        modelo = Pipeline([
            ("preprocessing", preprocessor),
            ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return modelo, report_dict, None
    except Exception as e:
        return None, None, str(e)


def img(name):
    path = os.path.join(GRAPHICS, name)
    return Image.open(path) if os.path.exists(path) else None


def show_img(name, caption=""):
    im = img(name)
    if im:
        st.image(im, caption=caption, use_container_width=True)
    else:
        st.info(f"Imagen no encontrada: {name}")


def vc(df, col, label="Registros"):
    """value_counts compatible con pandas antiguo y nuevo."""
    result = df.groupby(col).size().reset_index(name=label)
    result.columns = [col, label]
    return result


def ciudad_col(df):
    for c in ["municipio_de_residencia", "ciudad_de_residencia", "municipio", "ciudad"]:
        if c in df.columns:
            return c
    return None


def sabana_fig(fig, height=None):
    upd = dict(title_font_color=C_AZUL_SABANA, font_family="Arial",
               paper_bgcolor="white", plot_bgcolor="#F4F4F4")
    if height:
        upd["height"] = height
    return fig.update_layout(**upd)


# ── load ──────────────────────────────────────────────────────────────────────
try:
    asistencia, caract, perfiles = load_all()
    modelo_pred, feat_cols_pred = train_predictor(perfiles)
    modelo_logreg, report_logreg, error_logreg = train_logistic_model(caract, asistencia)
    ok = True
except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
    ok = False


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Proyecto Bienestar")
    st.markdown("**Universidad de la Sabana**")
    st.markdown("---")
    page = st.radio(
        "Sección",
        ["🏠 Inicio",
         "📊 Análisis Descriptivo",
         "👥 Perfiles Estudiantiles",
         "🤖 Modelos Predictivos",
         "🔍 Exploración Interactiva"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Programación Aplicada · 6.° Semestre · 2026")


# ══════════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.title("Proyecto Bienestar Estudiantil")
    st.markdown(
        "Análisis de la participación de estudiantes en programas de bienestar "
        "y su relación con el desempeño académico — Universidad de la Sabana (2015 – 2025)."
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if ok:
        n_asist   = len(asistencia)
        n_est_car = caract["codigo"].nunique()
        anos      = sorted(asistencia["ano"].dropna().unique())
        n_prog    = caract["programa_academico_principal"].nunique() \
                    if "programa_academico_principal" in caract.columns else "—"

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{n_asist:,}",              "Registros de asistencia"),
            (c2, f"{n_est_car:,}",            "Estudiantes caracterizados"),
            (c3, f"{int(anos[0])} – {int(anos[-1])}", "Periodo analizado"),
            (c4, str(n_prog),                 "Programas académicos"),
        ]:
            col.markdown(
                f"<div class='metric-box'><h2>{val}</h2><p>{label}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### ¿Qué encontrarás en este tablero?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**📊 Análisis Descriptivo**
- Distribución por género, ciudad y programa
- Evolución anual de la participación
- Promedios académicos (PAM y PCP)

**👥 Perfiles Estudiantiles**
- Segmentación mediante clustering K-Means
- Características de cada perfil
- Comparador interactivo de variables
""")
    with c2:
        st.markdown("""
**🤖 Modelos Predictivos**
- Regresión lineal, Random Forest y MLP
- Comparación de métricas de rendimiento
- **Simulador interactivo de predicción**

**🔍 Exploración Interactiva**
- Filtra por año, jefatura, sexo y programa
- Scatter individual con hover por estudiante
- Descarga los datos filtrados en CSV
""")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if ok:
        por_ano = asistencia.groupby("ano").size().reset_index(name="Registros")
        fig_home = px.bar(por_ano, x="ano", y="Registros", text_auto=True,
                          labels={"ano": "Año"},
                          title="Evolución anual de la participación en Bienestar",
                          template="plotly_white",
                          color="Registros", color_continuous_scale=SABANA_SCALE)
        fig_home.update_layout(coloraxis_showscale=False, title_font_color=C_AZUL_SABANA)
        st.plotly_chart(fig_home, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DESCRIPTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Análisis Descriptivo":
    st.title("Análisis Descriptivo")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not ok:
        st.warning("No se pudieron cargar los datos.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Participación", "Rendimiento Académico", "Demografía"])

    # ── Tab 1: Participación ──────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-title'>Participación en Bienestar</div>",
                    unsafe_allow_html=True)

        por_ano = asistencia.groupby("ano").size().reset_index(name="Registros")
        fig_ano = px.bar(por_ano, x="ano", y="Registros", text_auto=True,
                         labels={"ano": "Año"}, title="Evolución anual de asistencia",
                         template="plotly_white",
                         color="Registros", color_continuous_scale=SABANA_SCALE)
        sabana_fig(fig_ano.update_layout(coloraxis_showscale=False))

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_ano, use_container_width=True)
        with c2:
            if "jefatura" in asistencia.columns:
                por_jef = vc(asistencia, "jefatura").sort_values("Registros")
                fig_jef = px.bar(por_jef, x="Registros", y="jefatura", orientation="h",
                                 text_auto=True, template="plotly_white",
                                 labels={"jefatura": "Jefatura"},
                                 title="Registros por jefatura",
                                 color="Registros", color_continuous_scale=SABANA_SCALE)
                sabana_fig(fig_jef.update_layout(coloraxis_showscale=False), height=420)
                st.plotly_chart(fig_jef, use_container_width=True)

        if "estrategia_programa_o_servicio" in asistencia.columns:
            por_est = (vc(asistencia, "estrategia_programa_o_servicio", "Registros")
                       .sort_values("Registros").tail(15))
            fig_est = px.bar(por_est, x="Registros", y="estrategia_programa_o_servicio",
                             orientation="h", text_auto=True, template="plotly_white",
                             labels={"estrategia_programa_o_servicio": "Estrategia / Programa"},
                             title="Top 15 estrategias y programas de bienestar",
                             color="Registros", color_continuous_scale=SABANA_SCALE)
            sabana_fig(fig_est.update_layout(coloraxis_showscale=False), height=500)
            st.plotly_chart(fig_est, use_container_width=True)

        if "participacion_total" in perfiles.columns and PCP_COL in perfiles.columns:
            color_arg = ({"color": "perfil", "color_discrete_sequence": SABANA_QUALITATIVE}
                         if "perfil" in perfiles.columns else {})
            fig_scat = px.scatter(
                perfiles.sample(min(3000, len(perfiles)), random_state=42),
                x="participacion_total", y=PCP_COL, opacity=0.5,
                template="plotly_white",
                labels={"participacion_total": "Participación total", PCP_COL: "PCP"},
                title="Participación en Bienestar vs. Promedio semestral",
                **color_arg,
            )
            sabana_fig(fig_scat)
            st.plotly_chart(fig_scat, use_container_width=True)

    # ── Tab 2: Rendimiento Académico ──────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-title'>Rendimiento Académico</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if PAM_COL in caract.columns:
                fig_pam = px.histogram(caract, x=PAM_COL, nbins=40,
                                       labels={PAM_COL: "PAM"},
                                       title="Distribución del Promedio Acumulado (PAM)",
                                       template="plotly_white",
                                       color_discrete_sequence=[C_AZUL_CONTRASTE])
                sabana_fig(fig_pam)
                st.plotly_chart(fig_pam, use_container_width=True)
        with c2:
            if PCP_COL in caract.columns:
                fig_pcp = px.histogram(caract, x=PCP_COL, nbins=40,
                                       labels={PCP_COL: "PCP"},
                                       title="Distribución del Promedio Semestral (PCP)",
                                       template="plotly_white",
                                       color_discrete_sequence=[C_ACENTO])
                sabana_fig(fig_pcp)
                st.plotly_chart(fig_pcp, use_container_width=True)

        if PAM_COL in caract.columns and PCP_COL in caract.columns:
            src = perfiles if "perfil" in perfiles.columns else caract
            color_arg = ({"color": "perfil", "color_discrete_sequence": SABANA_QUALITATIVE}
                         if "perfil" in src.columns else {})
            fig_sc = px.scatter(src.sample(min(3000, len(src)), random_state=42),
                                x=PAM_COL, y=PCP_COL, opacity=0.5,
                                template="plotly_white",
                                labels={PAM_COL: "PAM", PCP_COL: "PCP"},
                                title="PAM vs. PCP por perfil estudiantil",
                                **color_arg)
            sabana_fig(fig_sc)
            st.plotly_chart(fig_sc, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if CRED_COL in caract.columns:
                fig_cred = px.histogram(caract, x=CRED_COL, nbins=30,
                                        labels={CRED_COL: "Créditos inscritos"},
                                        title="Distribución de créditos inscritos",
                                        template="plotly_white",
                                        color_discrete_sequence=[C_AZUL_CLARO])
                sabana_fig(fig_cred)
                st.plotly_chart(fig_cred, use_container_width=True)
        with c2:
            num_cols_corr = [c for c in [PAM_COL, PCP_COL, CRED_COL] if c in caract.columns]
            if len(num_cols_corr) >= 2:
                corr = caract[num_cols_corr].corr()
                fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                                     color_continuous_scale=SABANA_SCALE,
                                     title="Correlaciones entre variables académicas")
                sabana_fig(fig_heat)
                st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Demografía ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-title'>Demografía</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if "sexo" in caract.columns:
                por_sexo = vc(caract, "sexo", "Estudiantes")
                fig_donut = px.pie(por_sexo, names="sexo", values="Estudiantes",
                                   hole=0.45, template="plotly_white",
                                   title="Distribución por sexo",
                                   color_discrete_sequence=SABANA_QUALITATIVE)
                sabana_fig(fig_donut)
                st.plotly_chart(fig_donut, use_container_width=True)
        with c2:
            if "sexo" in caract.columns and PAM_COL in caract.columns:
                fig_viol = px.violin(caract, x="sexo", y=PAM_COL, box=True, color="sexo",
                                     labels={"sexo": "Sexo", PAM_COL: "PAM"},
                                     title="PAM por sexo", template="plotly_white",
                                     color_discrete_sequence=SABANA_QUALITATIVE)
                sabana_fig(fig_viol.update_layout(showlegend=False))
                st.plotly_chart(fig_viol, use_container_width=True)

        cc = ciudad_col(caract)
        if cc:
            top_ciudad = vc(caract, cc, "Estudiantes").sort_values("Estudiantes").tail(10)
            fig_ciudad = px.bar(top_ciudad, x="Estudiantes", y=cc, orientation="h",
                                text_auto=True, template="plotly_white",
                                labels={cc: "Ciudad"},
                                title="Top 10 ciudades de residencia",
                                color="Estudiantes", color_continuous_scale=SABANA_SCALE)
            sabana_fig(fig_ciudad.update_layout(coloraxis_showscale=False))
            st.plotly_chart(fig_ciudad, use_container_width=True)

        if "programa_academico_principal" in caract.columns:
            top_prog = (vc(caract, "programa_academico_principal", "Estudiantes")
                        .sort_values("Estudiantes").tail(10))
            fig_prog = px.bar(top_prog, x="Estudiantes", y="programa_academico_principal",
                              orientation="h", text_auto=True, template="plotly_white",
                              labels={"programa_academico_principal": "Programa"},
                              title="Top 10 programas académicos",
                              color="Estudiantes", color_continuous_scale=SABANA_SCALE)
            sabana_fig(fig_prog.update_layout(coloraxis_showscale=False))
            st.plotly_chart(fig_prog, use_container_width=True)

        if "sexo" in caract.columns and PCP_COL in caract.columns:
            fig_viol_pcp = px.violin(caract, x="sexo", y=PCP_COL, box=True, color="sexo",
                                     labels={"sexo": "Sexo", PCP_COL: "PCP"},
                                     title="PCP por sexo", template="plotly_white",
                                     color_discrete_sequence=SABANA_QUALITATIVE)
            sabana_fig(fig_viol_pcp.update_layout(showlegend=False))
            st.plotly_chart(fig_viol_pcp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PERFILES ESTUDIANTILES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Perfiles Estudiantiles":
    st.title("Perfiles Estudiantiles")
    st.markdown(
        "Segmentación de estudiantes mediante K-Means a partir de sus indicadores académicos "
        "y nivel de participación en bienestar."
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_img("clustering_seleccion_k.png", "Selección del número óptimo de clusters")
    with c2:
        show_img("perfil_tamanos.png", "Tamaño de cada perfil")

    st.markdown("### Caracterización de perfiles")
    c1, c2 = st.columns(2)
    with c1:
        show_img("perfil_radar.png", "Diagrama de radar por perfil")
    with c2:
        show_img("perfil_boxplots.png", "Boxplots por perfil")

    if ok:
        num_cols = [c for c in [PAM_COL, PCP_COL, CRED_COL, "participacion_total"]
                    if c in perfiles.columns]

        if "perfil" in perfiles.columns and num_cols:
            st.markdown("### Explorador interactivo de perfiles")

            resumen = (perfiles.groupby("perfil")[num_cols]
                       .agg(["mean", "median", "count"]).round(2))
            resumen.columns = [" — ".join(c) for c in resumen.columns]
            st.dataframe(resumen, use_container_width=True)

            st.markdown("#### Comparar variables")
            col_x, col_y = st.columns(2)
            eje_x = col_x.selectbox("Eje X", num_cols, index=0, key="px")
            eje_y = col_y.selectbox("Eje Y", num_cols,
                                    index=min(1, len(num_cols) - 1), key="py")

            fig_perf = px.scatter(
                perfiles.sample(min(3000, len(perfiles)), random_state=42),
                x=eje_x, y=eje_y, color="perfil", opacity=0.6,
                labels={eje_x: eje_x.split("(")[0].strip(),
                        eje_y: eje_y.split("(")[0].strip()},
                title=f"{eje_x.split('(')[0].strip()} vs. {eje_y.split('(')[0].strip()} por perfil",
                template="plotly_white",
                color_discrete_sequence=SABANA_QUALITATIVE,
            )
            sabana_fig(fig_perf)
            st.plotly_chart(fig_perf, use_container_width=True)

            var_box = st.selectbox("Variable para distribución por perfil", num_cols, key="pbox")
            fig_box = px.box(perfiles, x="perfil", y=var_box, color="perfil",
                             labels={"perfil": "Perfil",
                                     var_box: var_box.split("(")[0].strip()},
                             title=f"Distribución de {var_box.split('(')[0].strip()} por perfil",
                             template="plotly_white",
                             color_discrete_sequence=SABANA_QUALITATIVE)
            sabana_fig(fig_box.update_layout(showlegend=False))
            st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS PREDICTIVOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modelos Predictivos":
    st.title("Modelos Predictivos")
    st.markdown(
        "Se entrenaron tres modelos para predecir la participación estudiantil en bienestar: "
        "Regresión Lineal, Random Forest y Red Neuronal MLP."
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Comparación", "Random Forest", "Red Neuronal (MLP)", "🔵 Regresión Logística", "🎯 Simulador"]
    )

    with tab1:
        show_img("comparacion_modelos.png", "Comparación de modelos — métricas de error")
        show_img("regresion_coeficientes.png", "Coeficientes de regresión lineal")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            show_img("rf_importancia_variables.png", "Importancia de variables — Random Forest")
        with c2:
            show_img("shap_summary.png", "SHAP — interpretabilidad del modelo")

    with tab3:
        show_img("mlp_curvas_aprendizaje.png", "Curvas de aprendizaje — MLP")

    with tab4:
        st.markdown("<div class='section-title'>Regresión Logística</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "Clasifica la **actividad** de bienestar a la que asistirá un estudiante "
            "a partir de su sexo, ciudad de residencia y promedio acumulado (PAM)."
        )

        if error_logreg:
            st.error(f"No se pudo entrenar el modelo: {error_logreg}")
        elif report_logreg is None:
            st.warning("El modelo no está disponible.")
        else:
            metrics_rows = {
                clase: v for clase, v in report_logreg.items()
                if clase not in ("accuracy", "macro avg", "weighted avg")
                and isinstance(v, dict)
            }
            df_report = pd.DataFrame(metrics_rows).T[["precision", "recall", "f1-score", "support"]]
            df_report.index.name = "Actividad"
            df_report = df_report.sort_values("support", ascending=False)

            st.markdown("#### Reporte de clasificación por actividad")
            st.dataframe(df_report.style.format({
                "precision": "{:.2f}", "recall": "{:.2f}",
                "f1-score": "{:.2f}", "support": "{:.0f}",
            }), use_container_width=True)

            wa = report_logreg.get("weighted avg", {})
            ma = report_logreg.get("macro avg", {})
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"<div class='metric-box'><h2>{report_logreg.get('accuracy', 0):.2%}</h2>"
                "<p>Accuracy global</p></div>", unsafe_allow_html=True
            )
            c2.markdown(
                f"<div class='metric-box'><h2>{wa.get('f1-score', 0):.2f}</h2>"
                "<p>F1 ponderado</p></div>", unsafe_allow_html=True
            )
            c3.markdown(
                f"<div class='metric-box'><h2>{ma.get('f1-score', 0):.2f}</h2>"
                "<p>F1 macro</p></div>", unsafe_allow_html=True
            )

            st.markdown("#### F1-score por actividad (top 15)")
            top_f1 = df_report.head(15).reset_index()
            fig_lr = px.bar(
                top_f1, x="f1-score", y="Actividad", orientation="h",
                text_auto=".2f", template="plotly_white",
                labels={"f1-score": "F1-score"},
                color="f1-score", color_continuous_scale=SABANA_SCALE,
            )
            sabana_fig(fig_lr.update_layout(coloraxis_showscale=False), height=480)
            st.plotly_chart(fig_lr, use_container_width=True)

    with tab5:
        st.markdown("<div class='section-title'>Simulador de predicción</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "Ajusta los indicadores académicos del estudiante y obtén una estimación "
            "de su **participación total** esperada en programas de Bienestar."
        )

        if not ok or modelo_pred is None:
            st.warning("El modelo no pudo cargarse. Verifica que los datos estén disponibles.")
        else:
            col_meta = {
                PAM_COL:  ("Promedio Acumulado (PAM)",  0.1),
                PCP_COL:  ("Promedio Semestral (PCP)",  0.1),
                CRED_COL: ("Créditos inscritos",         1.0),
            }
            inputs = {}
            sliders = st.columns(len(feat_cols_pred))
            for i, col in enumerate(feat_cols_pred):
                label, step = col_meta.get(col, (col, 0.1))
                mn   = float(perfiles[col].min())
                mx   = float(perfiles[col].max())
                mean = float(round(perfiles[col].mean(), 2))
                inputs[col] = sliders[i].slider(label, min_value=mn, max_value=mx,
                                                 value=mean, step=step)

            pred = max(0.0, round(
                float(modelo_pred.predict([[inputs[c] for c in feat_cols_pred]])[0]), 1
            ))

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            res_col = st.columns([1, 2, 1])[1]
            with res_col:
                st.markdown(
                    f"<div class='metric-box'><h2>{pred}</h2>"
                    f"<p>Participaciones estimadas en Bienestar</p></div>",
                    unsafe_allow_html=True,
                )

            if "participacion_total" in perfiles.columns:
                pct = (perfiles["participacion_total"] <= pred).mean() * 100
                st.info(f"Este valor está por encima del **{pct:.0f}%** de los estudiantes en el histórico.")

                media = perfiles["participacion_total"].mean()
                fig_pred = px.histogram(
                    perfiles, x="participacion_total", nbins=40,
                    labels={"participacion_total": "Participación total"},
                    title="Tu estimación frente al histórico",
                    template="plotly_white",
                    color_discrete_sequence=[C_AZUL_CLARO],
                )
                fig_pred.add_vline(
                    x=pred,
                    line_color=C_ERROR if pred < media else C_ACENTO,
                    line_width=3, line_dash="dash",
                    annotation_text=f"Estimado: {pred}",
                    annotation_font_color=C_AZUL_SABANA,
                )
                sabana_fig(fig_pred)
                st.plotly_chart(fig_pred, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXPLORACIÓN INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploración Interactiva":
    st.title("Exploración Interactiva")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not ok:
        st.warning("No se pudieron cargar los datos para esta sección.")
        st.stop()

    # ── Filtros ───────────────────────────────────────────────────────────────
    st.markdown("### Filtros")
    fc1, fc2, fc3, fc4 = st.columns(4)

    anos_disp = sorted(asistencia["ano"].dropna().unique().astype(int))
    ano_range = fc1.slider("Rango de años",
                            min_value=int(anos_disp[0]), max_value=int(anos_disp[-1]),
                            value=(int(anos_disp[0]), int(anos_disp[-1])))

    jef_disp = (sorted(asistencia["jefatura"].dropna().unique())
                if "jefatura" in asistencia.columns else [])
    jef_sel = fc2.multiselect("Jefatura", jef_disp, default=jef_disp[:5])

    sexos_disp = sorted(caract["sexo"].dropna().unique()) if "sexo" in caract.columns else []
    sexo_sel = fc3.multiselect("Sexo", sexos_disp, default=sexos_disp)

    prog_disp = (sorted(caract["programa_academico_principal"].dropna().unique())
                 if "programa_academico_principal" in caract.columns else [])
    prog_sel = fc4.multiselect("Programa académico", prog_disp,
                                default=prog_disp[:10] if prog_disp else [])

    # Aplicar filtros
    asist_fil = asistencia[
        (asistencia["ano"] >= ano_range[0]) & (asistencia["ano"] <= ano_range[1])
    ]
    if jef_sel and "jefatura" in asist_fil.columns:
        asist_fil = asist_fil[asist_fil["jefatura"].isin(jef_sel)]

    caract_fil = caract.copy()
    if sexo_sel and "sexo" in caract.columns:
        caract_fil = caract_fil[caract_fil["sexo"].isin(sexo_sel)]
    if prog_sel and "programa_academico_principal" in caract.columns:
        caract_fil = caract_fil[caract_fil["programa_academico_principal"].isin(prog_sel)]

    n_est = caract_fil["codigo"].nunique() if "codigo" in caract_fil.columns else len(caract_fil)
    colr, cold = st.columns([3, 1])
    colr.markdown(
        f"**{len(asist_fil):,}** registros de asistencia · **{n_est:,}** estudiantes seleccionados"
    )
    cold.download_button(
        label="Descargar CSV",
        data=asist_fil.to_csv(index=False).encode("utf-8"),
        file_name="asistencia_filtrada.csv",
        mime="text/csv",
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Gráfica 1: por año ────────────────────────────────────────────────────
    st.markdown("#### Participación anual")
    if not asist_fil.empty:
        por_ano = asist_fil.groupby("ano").size().reset_index(name="Registros")
        fig1 = px.bar(por_ano, x="ano", y="Registros", text_auto=True,
                      labels={"ano": "Año"}, template="plotly_white",
                      color="Registros", color_continuous_scale=SABANA_SCALE)
        sabana_fig(fig1.update_layout(coloraxis_showscale=False))
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Sin datos con los filtros seleccionados.")

    # ── Gráfica 2: por jefatura ───────────────────────────────────────────────
    if not asist_fil.empty and "jefatura" in asist_fil.columns:
        st.markdown("#### Participación por jefatura")
        por_jef = vc(asist_fil, "jefatura").sort_values("Registros")
        fig2 = px.bar(por_jef, x="Registros", y="jefatura", orientation="h",
                      text_auto=True, template="plotly_white",
                      labels={"jefatura": "Jefatura"},
                      color="Registros", color_continuous_scale=SABANA_SCALE)
        sabana_fig(fig2.update_layout(coloraxis_showscale=False), height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Gráfica 3: scatter individual estudiantes ────────────────────────────
    if PAM_COL in caract_fil.columns and PCP_COL in caract_fil.columns:
        st.markdown("#### Distribución académica de estudiantes seleccionados")
        hover_cols = {c: True for c in ["codigo", "programa_academico_principal", CRED_COL]
                      if c in caract_fil.columns}
        color_col = "sexo" if "sexo" in caract_fil.columns else None
        fig3 = px.scatter(
            caract_fil.sample(min(3000, len(caract_fil)), random_state=42),
            x=PAM_COL, y=PCP_COL,
            color=color_col, opacity=0.55,
            hover_data=hover_cols,
            labels={PAM_COL: "PAM", PCP_COL: "PCP"},
            title="PAM vs. PCP — estudiantes seleccionados (hover para ver detalle)",
            template="plotly_white",
            color_discrete_sequence=SABANA_QUALITATIVE,
        )
        sabana_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Gráfica 4: distribución de promedios ──────────────────────────────────
    if PAM_COL in caract_fil.columns and PCP_COL in caract_fil.columns:
        st.markdown("#### Distribución de promedios")
        c1, c2 = st.columns(2)
        with c1:
            fig4a = px.histogram(caract_fil, x=PAM_COL, nbins=40,
                                 labels={PAM_COL: "PAM"}, title="Distribución PAM",
                                 template="plotly_white",
                                 color_discrete_sequence=[C_AZUL_CONTRASTE])
            sabana_fig(fig4a)
            st.plotly_chart(fig4a, use_container_width=True)
        with c2:
            fig4b = px.histogram(caract_fil, x=PCP_COL, nbins=40,
                                 labels={PCP_COL: "PCP"}, title="Distribución PCP",
                                 template="plotly_white",
                                 color_discrete_sequence=[C_ACENTO])
            sabana_fig(fig4b)
            st.plotly_chart(fig4b, use_container_width=True)

    # ── Gráfica 5: por sexo ───────────────────────────────────────────────────
    if "sexo" in caract_fil.columns:
        st.markdown("#### Distribución por sexo")
        por_sexo = vc(caract_fil, "sexo", "Estudiantes")
        fig5 = px.pie(por_sexo, names="sexo", values="Estudiantes",
                      hole=0.4, template="plotly_white",
                      color_discrete_sequence=SABANA_QUALITATIVE)
        sabana_fig(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Vista previa de datos ─────────────────────────────────────────────────
    with st.expander("Ver datos de asistencia filtrados (primeras 500 filas)"):
        st.dataframe(asist_fil.head(500), use_container_width=True)
