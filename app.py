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

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.metric-box {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    border-radius: 12px; padding: 18px 14px; text-align: center;
    color: white; margin-bottom: 8px;
}
.metric-box h2 { font-size: 2rem; margin: 0; }
.metric-box p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }
.section-title { font-size: 1.5rem; font-weight: 700; color: #1e3a5f; margin-bottom: 4px; }
.divider { border: none; border-top: 2px solid #e0e0e0; margin: 12px 0 20px; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "Data")
GRAPHICS = os.path.join(BASE, "src", "graphics")


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos…")
def load_all():
    asist_old = pd.read_csv(os.path.join(DATA, "asistencia_2015_2022_limpia.csv"))
    asist_new = pd.read_csv(os.path.join(DATA, "asistencia_2023_2025_limpia.csv"))
    caract     = pd.read_csv(os.path.join(DATA, "caracterizacion_limpia.csv"))
    perfiles   = pd.read_csv(os.path.join(DATA, "perfiles_estudiantiles.csv"))

    # Unify attendance tables
    if "ano" not in asist_new.columns:
        asist_new["ano"] = asist_new["periodo"].astype(str).str[:4].astype(int, errors="ignore")
    common = ["ano", "periodo", "jefatura", "estrategia_programa_o_servicio", "actividad", "codigo"]
    asist_old_sub = asist_old[[c for c in common if c in asist_old.columns]]
    asist_new_sub = asist_new[[c for c in common if c in asist_new.columns]]
    asistencia = pd.concat([asist_old_sub, asist_new_sub], ignore_index=True)

    return asistencia, caract, perfiles


def img(name):
    path = os.path.join(GRAPHICS, name)
    return Image.open(path) if os.path.exists(path) else None


def show_img(name, caption="", use_col_width=True):
    im = img(name)
    if im:
        st.image(im, caption=caption, use_container_width=use_col_width)
    else:
        st.info(f"Imagen no encontrada: {name}")


# ── load ──────────────────────────────────────────────────────────────────────
try:
    asistencia, caract, perfiles = load_all()
    ok = True
except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
    ok = False


# ── sidebar navigation ────────────────────────────────────────────────────────
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
        n_prog    = caract["programa_academico_principal"].nunique() if "programa_academico_principal" in caract.columns else "—"

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{n_asist:,}", "Registros de asistencia"),
            (c2, f"{n_est_car:,}", "Estudiantes caracterizados"),
            (c3, f"{int(anos[0])} – {int(anos[-1])}", "Periodo analizado"),
            (c4, str(n_prog), "Programas académicos"),
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
- Diagramas de radar y dispersión
""")
    with c2:
        st.markdown("""
**🤖 Modelos Predictivos**
- Regresión lineal, Random Forest y MLP
- Comparación de métricas de rendimiento
- Importancia de variables (SHAP)

**🔍 Exploración Interactiva**
- Filtra por año, jefatura y sexo
- Genera gráficas dinámicas con tus selecciones
""")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    col_img = st.columns([1, 2, 1])[1]
    with col_img:
        show_img("bar_evolucion_anual.png", "Evolución anual de la participación")


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DESCRIPTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Análisis Descriptivo":
    st.title("Análisis Descriptivo")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Participación", "Rendimiento Académico", "Demografía"])

    with tab1:
        st.markdown("<div class='section-title'>Participación en Bienestar</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show_img("bar_evolucion_anual.png", "Evolución anual")
        with c2:
            show_img("bar_jefatura.png", "Participación por jefatura")
        c1, c2 = st.columns(2)
        with c1:
            show_img("bar_grupo_academico.png", "Por grupo académico")
        with c2:
            show_img("participacion_vs_pcp.png", "Participación vs. Promedio semestral")

    with tab2:
        st.markdown("<div class='section-title'>Rendimiento Académico</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show_img("hist_promedio_acumulado_programa_principalpam.png", "Distribución PAM")
        with c2:
            show_img("hist_promedio_semestral_programa_principalpcp.png", "Distribución PCP")
        c1, c2 = st.columns(2)
        with c1:
            show_img("scatter_pam_vs_pcp.png", "PAM vs. PCP")
        with c2:
            show_img("box_promedios_academicos.png", "Boxplot promedios")
        c1, c2 = st.columns(2)
        with c1:
            show_img("hist_creditos_inscritos.png", "Créditos inscritos")
        with c2:
            show_img("heatmap_correlaciones.png", "Correlaciones")

    with tab3:
        st.markdown("<div class='section-title'>Demografía</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show_img("donut_sexo.png", "Distribución por sexo")
        with c2:
            show_img("participacion_por_genero.png", "Participación por género")
        c1, c2 = st.columns(2)
        with c1:
            show_img("bar_top10_ciudad.png", "Top 10 ciudades")
        with c2:
            show_img("participacion_por_municipio.png", "Participación por municipio")
        show_img("bar_top10_programa.png", "Top 10 programas académicos")
        st.markdown("#### Promedios académicos por sexo")
        c1, c2 = st.columns(2)
        with c1:
            show_img("violin_promedio_acumulado_programa_principalpam_por_sexo.png", "PAM por sexo")
        with c2:
            show_img("violin_promedio_semestral_programa_principalpcp_por_sexo.png", "PCP por sexo")


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

    st.markdown("### Relación con el rendimiento académico")
    c1, c2 = st.columns(2)
    with c1:
        show_img("perfil_scatter_pam_participacion.png", "PAM vs. Participación por perfil")
    with c2:
        show_img("perfil_scatter_pcp_participacion.png", "PCP vs. Participación por perfil")

    if ok:
        st.markdown("### Resumen estadístico por perfil")
        num_cols = [
            "promedio_acumulado_programa_principal(pam)",
            "promedio_semestral_programa_principal(pcp)",
            "creditos_inscritos",
            "participacion_total",
        ]
        available = [c for c in num_cols if c in perfiles.columns]
        if "perfil" in perfiles.columns and available:
            resumen = (
                perfiles.groupby("perfil")[available]
                .agg(["mean", "median", "count"])
                .round(2)
            )
            resumen.columns = [" — ".join(c) for c in resumen.columns]
            st.dataframe(resumen, use_container_width=True)

            if "participacion_total" in perfiles.columns:
                fig = px.box(
                    perfiles,
                    x="perfil",
                    y="participacion_total",
                    color="perfil",
                    title="Distribución de participación total por perfil",
                    labels={"participacion_total": "Participación total", "perfil": "Perfil"},
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)


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

    tab1, tab2, tab3 = st.tabs(["Comparación", "Random Forest", "Red Neuronal (MLP)"])

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


# ══════════════════════════════════════════════════════════════════════════════
# EXPLORACIÓN INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploración Interactiva":
    st.title("Exploración Interactiva")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not ok:
        st.warning("No se pudieron cargar los datos para esta sección.")
        st.stop()

    # ── filters ──────────────────────────────────────────────────────────────
    st.markdown("### Filtros")
    fc1, fc2, fc3 = st.columns(3)

    anos_disp = sorted(asistencia["ano"].dropna().unique().astype(int))
    anos_sel = fc1.multiselect("Año", anos_disp, default=anos_disp[-5:])

    jefaturas_disp = sorted(asistencia["jefatura"].dropna().unique())
    jef_sel = fc2.multiselect("Jefatura", jefaturas_disp, default=jefaturas_disp[:5])

    sexos_disp = sorted(caract["sexo"].dropna().unique()) if "sexo" in caract.columns else []
    sexo_sel = fc3.multiselect("Sexo (caracterización)", sexos_disp, default=sexos_disp)

    asist_fil = asistencia[
        asistencia["ano"].isin(anos_sel) & asistencia["jefatura"].isin(jef_sel)
    ]
    caract_fil = caract.copy()
    if sexo_sel and "sexo" in caract.columns:
        caract_fil = caract_fil[caract_fil["sexo"].isin(sexo_sel)]

    st.markdown(
        f"**{len(asist_fil):,}** registros de asistencia · "
        f"**{caract_fil['codigo'].nunique():,}** estudiantes seleccionados"
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── chart 1: participation by year ───────────────────────────────────────
    st.markdown("#### Participación anual")
    if not asist_fil.empty:
        por_ano = asist_fil.groupby("ano").size().reset_index(name="registros")
        fig1 = px.bar(
            por_ano, x="ano", y="registros",
            text_auto=True,
            labels={"ano": "Año", "registros": "Registros de asistencia"},
            template="plotly_white",
            color="registros",
            color_continuous_scale="Blues",
        )
        fig1.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Sin datos con los filtros seleccionados.")

    # ── chart 2: participation by jefatura ────────────────────────────────────
    st.markdown("#### Participación por jefatura")
    if not asist_fil.empty:
        por_jef = (
            asist_fil.groupby("jefatura").size()
            .reset_index(name="registros")
            .sort_values("registros", ascending=True)
        )
        fig2 = px.bar(
            por_jef, x="registros", y="jefatura", orientation="h",
            text_auto=True,
            labels={"jefatura": "Jefatura", "registros": "Registros"},
            template="plotly_white",
            color="registros",
            color_continuous_scale="Teal",
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # ── chart 3: academic distribution ────────────────────────────────────────
    st.markdown("#### Distribución de promedios (caracterización filtrada)")
    pam_col = "promedio_acumulado_programa_principal(pam)"
    pcp_col = "promedio_semestral_programa_principal(pcp)"
    if pam_col in caract_fil.columns and pcp_col in caract_fil.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.histogram(
                caract_fil, x=pam_col, nbins=40,
                labels={pam_col: "PAM"},
                title="Distribución PAM",
                template="plotly_white",
                color_discrete_sequence=["#2d6a9f"],
            )
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.histogram(
                caract_fil, x=pcp_col, nbins=40,
                labels={pcp_col: "PCP"},
                title="Distribución PCP",
                template="plotly_white",
                color_discrete_sequence=["#1e8e7e"],
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── chart 4: sex breakdown ─────────────────────────────────────────────────
    if "sexo" in caract_fil.columns:
        st.markdown("#### Distribución por sexo (caracterización filtrada)")
        por_sexo = caract_fil["sexo"].value_counts().reset_index()
        por_sexo.columns = ["sexo", "count"]
        fig5 = px.pie(
            por_sexo, names="sexo", values="count",
            hole=0.4,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── raw data preview ──────────────────────────────────────────────────────
    with st.expander("Ver datos de asistencia filtrados (primeras 500 filas)"):
        st.dataframe(asist_fil.head(500), use_container_width=True)
