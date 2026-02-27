"""
Dashboard Interactivo: Producción de Arroz en Colombia
Análisis Exploratorio de Datos (EDA) - Enfoque: De Macro a Específico
Grupo 2 MAD
Fecha de creación: 24 de febrero de 2026
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# ==================== CONFIGURACIÓN DE LA PÁGINA ====================
st.set_page_config(
    page_title="Dashboard Arroz Colombia",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ESTILOS CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F5E9, #C8E6C9);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #558B2F;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976D2;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CARGA DE DATOS ====================
@st.cache_data
def load_data():
    """Carga y preprocesa los datos de arroz desde Dropbox"""
    url = 'https://www.dropbox.com/scl/fi/e3iwe3z3jszouxues5bai/data21022026.csv?rlkey=fb1ex5sbf7yz4p0im8gfiziwm&st=va67mghf&dl=1'
    
    # Cargar dataset completo
    df_completo = pd.read_csv(url, delimiter=';', encoding='utf-8')
    
    # Limpiar nombres de columnas
    df_completo.columns = df_completo.columns.str.strip().str.replace('\n', ' ')
    
    # Filtrar solo ARROZ
    df = df_completo[
        (df_completo['GRUPO  DE CULTIVO'].str.strip().str.upper() == 'CEREALES') &
        (df_completo['CULTIVO'].str.strip().str.upper() == 'ARROZ')
    ].copy()
    
    # Renombrar columnas
    column_mapping = {
        'CÓD.  DEP.': 'cod_dep',
        'DEPARTAMENTO': 'departamento',
        'CÓD. MUN.': 'cod_mun',
        'MUNICIPIO': 'municipio',
        'GRUPO  DE CULTIVO': 'grupo_cultivo',
        'SUBGRUPO  DE CULTIVO': 'subgrupo_cultivo',
        'CULTIVO': 'cultivo',
        'DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO': 'sistema_productivo',
        'AÑO': 'año',
        'PERIODO': 'periodo',
        'Área Sembrada (ha)': 'area_sembrada',
        'Área Cosechada (ha)': 'area_cosechada',
        'Producción (t)': 'produccion',
        'Rendimiento (t/ha)': 'rendimiento',
        'ESTADO FISICO PRODUCCION': 'estado_fisico',
        'NOMBRE  CIENTIFICO': 'nombre_cientifico',
        'CICLO DE CULTIVO': 'ciclo_cultivo'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Convertir columnas numéricas
    numeric_columns = ['area_sembrada', 'area_cosechada', 'produccion', 'rendimiento']
    for col in numeric_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)
    
    return df

# Cargar datos
with st.spinner('Cargando datos de arroz...'):
    df = load_data()

# ==================== SIDEBAR - FILTROS INTERACTIVOS ====================
st.sidebar.markdown("## Filtros Interactivos")
st.sidebar.markdown("---")

# Filtro de Departamentos
departamentos_disponibles = sorted(df['departamento'].unique())
departamentos_seleccionados = st.sidebar.multiselect(
    "Selecciona Departamentos:",
    options=departamentos_disponibles,
    default=departamentos_disponibles,  # Todos seleccionados por defecto
    help="Selecciona departamentos para filtrar"
)

st.sidebar.markdown("---")

# ==================== APLICAR FILTROS ====================
df_filtrado = df.copy()

# Aplicar filtro de departamentos
if departamentos_seleccionados:
    df_filtrado = df_filtrado[df_filtrado['departamento'].isin(departamentos_seleccionados)]

# Validar que hay datos después de filtrar
if df_filtrado.empty:
    st.error("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta tus selecciones.")
    st.stop()

# ==================== HEADER ====================
st.markdown('<div class="main-header">Dashboard: Producción de Arroz en Colombia</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>Contexto del Dashboard</h3>
    <p><strong>Objetivo:</strong> Analizar la producción de arroz en Colombia utilizando datos de las Evaluaciones Agropecuarias Municipales (EVA) 
    del Ministerio de Agricultura.</p>
    <p><strong>Preguntas:</strong></p>
    <ul>
        <li>¿Cómo ha evolucionado la producción de arroz a lo largo del tiempo en Colombia?</li>
        <li>¿Cuál es la distribución geográfica de la producción de arroz en Colombia?</li>
        <li>¿Qué sistemas productivos se utilizan y cuál es su eficiencia?</li>
        <li>¿Cómo se comparan los rendimientos entre regiones?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==================== FILTROS ADICIONALES ====================
st.markdown("## Filtros de Análisis")

col_filtro1, col_filtro2, col_filtro3 = st.columns(3)

with col_filtro1:
    # Filtro de Año específico
    if 'año' in df_filtrado.columns and df_filtrado['año'].notna().any():
        años_filtro = sorted(df_filtrado['año'].dropna().unique(), reverse=True)
        año_analisis = st.selectbox(
            "Selecciona Año para Análisis:",
            options=['Todos'] + [int(a) for a in años_filtro],
            index=0,  # Índice 0 = "Todos"
            help="Filtra los datos por un año específico. Por defecto muestra todos los años."
        )
    else:
        año_analisis = 'Todos'

with col_filtro2:
    # Filtro de Departamentos
    departamentos_filtro = sorted(df_filtrado['departamento'].unique())
    departamento_analisis = st.selectbox(
        "Selecciona Departamento:",
        options=['Todos'] + departamentos_filtro,
        index=0,
        help="Filtra los datos por un departamento específico. Por defecto muestra todos."
    )

with col_filtro3:
    # Filtro de Municipios (depende del departamento seleccionado)
    if departamento_analisis != 'Todos':
        municipios_filtro = sorted(df_filtrado[df_filtrado['departamento'] == departamento_analisis]['municipio'].unique())
    else:
        municipios_filtro = sorted(df_filtrado['municipio'].unique())
    
    municipio_analisis = st.selectbox(
        "Selecciona Municipio:",
        options=['Todos'] + municipios_filtro,
        index=0,
        help="Filtra los datos por un municipio específico. Por defecto muestra todos."
    )

# Aplicar filtros
if año_analisis != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['año'] == año_analisis]

if departamento_analisis != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['departamento'] == departamento_analisis]

if municipio_analisis != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['municipio'] == municipio_analisis]

# Validar que hay datos después de aplicar filtros adicionales
if df_filtrado.empty:
    st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta tus selecciones.")
    st.stop()

st.markdown("---")

# ==================== MÉTRICAS PRINCIPALES ====================
st.markdown("## Métricas Principales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_produccion = df_filtrado['produccion'].sum() / 1_000_000  # Millones de toneladas
    st.metric(
        label="Producción Total",
        value=f"{total_produccion:.2f} M ton",
        help="Producción total de cereales en millones de toneladas"
    )

with col2:
    area_total = df_filtrado['area_sembrada'].sum() / 1_000  # Miles de hectáreas
    st.metric(
        label="Área Sembrada",
        value=f"{area_total:.1f}K ha",
        help="Área total sembrada en miles de hectáreas"
    )

with col3:
    rendimiento_promedio = df_filtrado['rendimiento'].mean()
    st.metric(
        label="Rendimiento Promedio",
        value=f"{rendimiento_promedio:.2f} t/ha",
        help="Rendimiento promedio en toneladas por hectárea"
    )

with col4:
    num_municipios = df_filtrado['municipio'].nunique()
    st.metric(
        label="Municipios",
        value=f"{num_municipios}",
        help="Número de municipios productores"
    )

st.markdown("---")

# ==================== SECCIÓN 1: ANÁLISIS MACRO - VISIÓN GENERAL ====================
st.markdown("# SECCIÓN 1: Análisis Macro - Visión General")
st.markdown("---")

# ==================== VISUALIZACIÓN 1.1: EVOLUCIÓN TEMPORAL ====================
st.markdown("## 1.1 Evolución Temporal de la Producción de Arroz")

if 'año' in df_filtrado.columns and df_filtrado['año'].notna().sum() > 0:
    # Producción por año
    produccion_año = df_filtrado.groupby('año').agg({
        'produccion': 'sum',
        'area_sembrada': 'sum',
        'rendimiento': 'mean'
    }).reset_index()
    
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Producción y Área Sembrada por Año', 'Rendimiento Promedio por Año'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Gráfico 1: Producción y Área
    fig1.add_trace(
        go.Scatter(x=produccion_año['año'], y=produccion_año['produccion']/1000,
                   name='Producción (K ton)', line=dict(color='green', width=3),
                   mode='lines+markers+text',
                   text=[f"{val/1000:.1f}" for val in produccion_año['produccion']],
                   textposition='top center',
                   textfont=dict(size=10)),
        row=1, col=1, secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=produccion_año['año'], y=produccion_año['area_sembrada']/1000,
                   name='Área Sembrada (K ha)', line=dict(color='orange', width=3, dash='dash'),
                   mode='lines+markers+text',
                   text=[f"{val/1000:.1f}" for val in produccion_año['area_sembrada']],
                   textposition='bottom center',
                   textfont=dict(size=10)),
        row=1, col=1, secondary_y=True
    )
    
    # Gráfico 2: Rendimiento
    fig1.add_trace(
        go.Scatter(x=produccion_año['año'], y=produccion_año['rendimiento'],
                   name='Rendimiento (t/ha)', line=dict(color='blue', width=3),
                   fill='tozeroy',
                   mode='lines+markers+text',
                   text=[f"{val:.2f}" for val in produccion_año['rendimiento']],
                   textposition='top center',
                   textfont=dict(size=10)),
        row=2, col=1
    )
    
    fig1.update_xaxes(title_text="Año", row=2, col=1)
    fig1.update_yaxes(title_text="Producción (K ton)", row=1, col=1, secondary_y=False)
    fig1.update_yaxes(title_text="Área (K ha)", row=1, col=1, secondary_y=True)
    fig1.update_yaxes(title_text="Rendimiento (t/ha)", row=2, col=1)
    
    fig1.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Cálculo de tendencias
    años_total = produccion_año['año'].nunique()
    prod_inicio = produccion_año.iloc[0]['produccion']
    prod_fin = produccion_año.iloc[-1]['produccion']
    cambio_porcentual = ((prod_fin - prod_inicio) / prod_inicio * 100) if prod_inicio > 0 else 0
else:
    st.warning("No hay datos temporales disponibles.")

st.markdown("---")

# ==================== VISUALIZACIÓN 1.2: DISTRIBUCIÓN GEOGRÁFICA ====================
st.markdown("## 1.2 Distribución Geográfica por Departamento")

col1, col2 = st.columns([2, 1])

with col1:
    # Producción por departamento - Ranking completo (mayor a menor, de arriba hacia abajo)
    produccion_dep = df_filtrado.groupby('departamento')['produccion'].sum().sort_values(ascending=False)
    
    # Calcular porcentaje acumulado
    produccion_dep_acum = produccion_dep.cumsum()
    porcentaje_acum = (produccion_dep_acum / produccion_dep.sum() * 100).round(1)
    
    # Crear DataFrame para el gráfico
    df_grafico = pd.DataFrame({
        'departamento': produccion_dep.index,
        'produccion': produccion_dep.values,
        'porcentaje_acumulado': porcentaje_acum.values
    }).sort_values('produccion', ascending=True)  # Invertir para gráfico horizontal
    
    fig2 = px.bar(
        df_grafico,
        x='produccion',
        y='departamento',
        orientation='h',
        title='Ranking Completo de Departamentos por Producción de Arroz',
        labels={'produccion': 'Producción (toneladas)', 'departamento': 'Departamento'},
        color='produccion',
        color_continuous_scale='Greens',
        text='porcentaje_acumulado'
    )
    fig2.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    fig2.update_layout(
        height=800, 
        showlegend=False,
        xaxis_title='Producción (toneladas) | % Acumulado'
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("### Interpretación Macro")
    produccion_dep_desc = produccion_dep.sort_values(ascending=False)  # Para mostrar estadísticas correctas
    st.markdown(f"""
    - **Departamentos productores:** {df_filtrado['departamento'].nunique()}
    - **Líder en producción:** {produccion_dep_desc.index[0]}
    - **Producción líder:** {produccion_dep_desc.values[0]/1000:.1f}K ton
    - **Concentración:** Los top 5 representan el {(produccion_dep_desc.head(5).sum() / produccion_dep_desc.sum() * 100):.1f}% del total
    
    Esta visión macro muestra la distribución espacial de la producción arrocera en Colombia.
    """)

st.markdown("---")

# ==================== VISUALIZACIÓN 1.3: MAPA DE CALOR GEOGRÁFICO ====================
st.markdown("## 1.3 Mapa de Calor de Producción por Departamento")

# Agrupar producción por municipio y departamento
produccion_municipios = df_filtrado.groupby(['departamento', 'municipio']).agg({
    'produccion': 'sum',
    'area_sembrada': 'sum',
    'rendimiento': 'mean'
}).reset_index()

# Crear un identificador único para cada municipio
produccion_municipios['municipio_completo'] = (
    produccion_municipios['municipio'] + ', ' + produccion_municipios['departamento']
)

# Ordenar por producción
produccion_municipios = produccion_municipios.sort_values('produccion', ascending=False)

# Agrupar por departamento para el mapa
produccion_departamentos = df_filtrado.groupby('departamento').agg({
    'produccion': 'sum',
    'area_sembrada': 'sum',
    'rendimiento': 'mean'
}).reset_index()

# Coordenadas aproximadas de los departamentos de Colombia (capitales)
coordenadas_dept = {
    'ANTIOQUIA': {'lat': 6.2518, 'lon': -75.5636},
    'ATLANTICO': {'lat': 10.9639, 'lon': -74.7964},
    'BOLIVAR': {'lat': 10.3910, 'lon': -75.4794},
    'BOYACA': {'lat': 5.4545, 'lon': -73.3625},
    'CALDAS': {'lat': 5.0689, 'lon': -75.5174},
    'CAQUETA': {'lat': 1.6144, 'lon': -75.6062},
    'CAUCA': {'lat': 2.4448, 'lon': -76.6147},
    'CESAR': {'lat': 10.4631, 'lon': -73.2532},
    'CORDOBA': {'lat': 8.7479, 'lon': -75.8814},
    'CUNDINAMARCA': {'lat': 4.7110, 'lon': -74.0721},
    'CHOCO': {'lat': 5.6983, 'lon': -76.6583},
    'HUILA': {'lat': 2.9273, 'lon': -75.2819},
    'LA GUAJIRA': {'lat': 11.5448, 'lon': -72.9072},
    'MAGDALENA': {'lat': 11.2408, 'lon': -74.1990},
    'META': {'lat': 4.1420, 'lon': -73.6266},
    'NARIÑO': {'lat': 1.2136, 'lon': -77.2811},
    'NORTE DE SANTANDER': {'lat': 7.8939, 'lon': -72.5078},
    'QUINDIO': {'lat': 4.4610, 'lon': -75.6674},
    'RISARALDA': {'lat': 4.8087, 'lon': -75.6906},
    'SANTANDER': {'lat': 7.1254, 'lon': -73.1198},
    'SUCRE': {'lat': 9.2985, 'lon': -75.3976},
    'TOLIMA': {'lat': 4.4389, 'lon': -75.2322},
    'VALLE DEL CAUCA': {'lat': 3.4516, 'lon': -76.5320},
    'ARAUCA': {'lat': 7.0904, 'lon': -70.7619},
    'CASANARE': {'lat': 5.3356, 'lon': -72.3959},
    'PUTUMAYO': {'lat': 0.8667, 'lon': -76.8500},
    'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA': {'lat': 12.5847, 'lon': -81.7006},
    'AMAZONAS': {'lat': -4.2051, 'lon': -69.9406},
    'GUAINIA': {'lat': 2.5833, 'lon': -67.9167},
    'GUAVIARE': {'lat': 2.5700, 'lon': -72.6367},
    'VAUPES': {'lat': 0.8333, 'lon': -70.8167},
    'VICHADA': {'lat': 4.4167, 'lon': -69.2833}
}

# Agregar coordenadas a los datos de producción
produccion_departamentos['lat'] = produccion_departamentos['departamento'].map(
    lambda x: coordenadas_dept.get(x.upper(), {}).get('lat', None)
)
produccion_departamentos['lon'] = produccion_departamentos['departamento'].map(
    lambda x: coordenadas_dept.get(x.upper(), {}).get('lon', None)
)

# Filtrar departamentos sin coordenadas
produccion_departamentos_mapa = produccion_departamentos.dropna(subset=['lat', 'lon'])

# Crear mapa estilo Google Maps
produccion_departamentos_mapa['produccion_label'] = produccion_departamentos_mapa['produccion'].apply(lambda x: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}")

fig_mapa = px.scatter_mapbox(
    produccion_departamentos_mapa,
    lat='lat',
    lon='lon',
    size='produccion',
    color='produccion',
    text='produccion_label',
    hover_name='departamento',
    hover_data={
        'produccion': ':.2f',
        'area_sembrada': ':.2f',
        'rendimiento': ':.2f',
        'lat': False,
        'lon': False,
        'produccion_label': False
    },
    color_continuous_scale='YlGn',
    size_max=50,
    title='Mapa de Producción de Arroz por Departamento en Colombia',
    labels={'produccion': 'Producción (ton)'},
    zoom=5,
    center={"lat": 4.5709, "lon": -74.2973}
)
fig_mapa.update_traces(textfont=dict(size=9, color='black'), textposition='top center')

# Configurar estilo del mapa (minimalista para mejor visualización)
fig_mapa.update_layout(
    mapbox_style="carto-positron",  # Estilo limpio y minimalista
    height=700,
    margin={"r":0,"t":40,"l":0,"b":0}
)

st.plotly_chart(fig_mapa, use_container_width=True)

# Tabla con Top 10 departamentos
col_top1, col_top2 = st.columns(2)

with col_top1:
    st.markdown("### Top 10 Departamentos por Producción")
    top_departamentos = produccion_departamentos_mapa.sort_values('produccion', ascending=False).head(10)[['departamento', 'produccion', 'area_sembrada', 'rendimiento']].copy()
    top_departamentos['produccion'] = (top_departamentos['produccion'] / 1000).round(2)
    top_departamentos['area_sembrada'] = (top_departamentos['area_sembrada'] / 1000).round(2)
    top_departamentos['rendimiento'] = top_departamentos['rendimiento'].round(2)
    top_departamentos.columns = ['Departamento', 'Producción (K ton)', 'Área (K ha)', 'Rendimiento (t/ha)']
    st.dataframe(top_departamentos, hide_index=True, use_container_width=True)

with col_top2:
    st.markdown("### Estadísticas por Departamento")
    total_departamentos = len(produccion_departamentos)
    prod_top10_dept = produccion_departamentos.nlargest(10, 'produccion')['produccion'].sum()
    prod_total_dept = produccion_departamentos['produccion'].sum()
    concentracion_dept = (prod_top10_dept / prod_total_dept * 100) if prod_total_dept > 0 else 0
    
    st.markdown(f"""
    - **Total de departamentos productores:** {total_departamentos}
    - **Departamento líder:** {produccion_departamentos.nlargest(1, 'produccion')['departamento'].iloc[0]}
    - **Producción del líder:** {produccion_departamentos['produccion'].max()/1000:.2f}K ton
    - **Concentración Top 10:** {concentracion_dept:.1f}% de la producción total
    - **Mayor rendimiento departamental:** {produccion_departamentos.loc[produccion_departamentos['rendimiento'].idxmax(), 'departamento']} ({produccion_departamentos['rendimiento'].max():.2f} t/ha)
    
    El mapa de calor muestra la concentración geográfica de la producción arrocera por departamento.
    """)

st.markdown("---")

# ==================== SECCIÓN 2: ANÁLISIS DE EFICIENCIA Y PÉRDIDAS ====================
st.markdown("# SECCIÓN 2: Análisis de Eficiencia y Pérdidas Agrícolas")
st.markdown("---")

# ==================== VISUALIZACIÓN 2.1: EFICIENCIA DE COSECHA ====================
st.markdown("## 2.1 Análisis de Pérdidas: Área Sembrada vs Cosechada")

if 'area_cosechada' in df_filtrado.columns and df_filtrado['area_cosechada'].notna().sum() > 0:
    # Calcular eficiencia por departamento
    eficiencia_dept = df_filtrado.groupby('departamento').agg({
        'area_sembrada': 'sum',
        'area_cosechada': 'sum',
        'produccion': 'sum'
    }).reset_index()
    
    eficiencia_dept['perdida'] = eficiencia_dept['area_sembrada'] - eficiencia_dept['area_cosechada']
    eficiencia_dept['eficiencia_%'] = (eficiencia_dept['area_cosechada'] / eficiencia_dept['area_sembrada'] * 100).round(2)
    eficiencia_dept = eficiencia_dept.sort_values('perdida', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras apiladas
        fig_efic1 = go.Figure()
        
        fig_efic1.add_trace(go.Bar(
            y=eficiencia_dept['departamento'].head(15),
            x=eficiencia_dept['area_cosechada'].head(15),
            name='Área Cosechada',
            orientation='h',
            marker=dict(color='green'),
            text=[f"{val/1000:.1f}K" for val in eficiencia_dept['area_cosechada'].head(15)],
            textposition='inside'
        ))
        
        fig_efic1.add_trace(go.Bar(
            y=eficiencia_dept['departamento'].head(15),
            x=eficiencia_dept['perdida'].head(15),
            name='Área No Cosechada (Pérdida)',
            orientation='h',
            marker=dict(color='red'),
            text=[f"{val/1000:.1f}K" if val >= 1000 else f"{val:.0f}" for val in eficiencia_dept['perdida'].head(15)],
            textposition='inside'
        ))
        
        fig_efic1.update_layout(
            barmode='stack',
            title='Top 15 Departamentos: Área Sembrada vs Cosechada',
            xaxis_title='Hectáreas',
            yaxis_title='Departamento',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_efic1, use_container_width=True)
    
    with col2:
        # Gráfico de eficiencia porcentual
        fig_efic2 = px.bar(
            eficiencia_dept.head(15),
            y='departamento',
            x='eficiencia_%',
            orientation='h',
            title='Top 15: Eficiencia de Cosecha (%)',
            labels={'eficiencia_%': 'Eficiencia (%)', 'departamento': 'Departamento'},
            color='eficiencia_%',
            color_continuous_scale='RdYlGn',
            range_color=[80, 100],
            text='eficiencia_%'
        )
        fig_efic2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_efic2.update_layout(height=600)
        st.plotly_chart(fig_efic2, use_container_width=True)
    
    # Métricas de eficiencia
    perdida_total = eficiencia_dept['perdida'].sum()
    area_sembrada_total = eficiencia_dept['area_sembrada'].sum()
    eficiencia_promedio = (eficiencia_dept['area_cosechada'].sum() / area_sembrada_total * 100)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Área Total Sembrada", f"{area_sembrada_total/1000:.1f}K ha")
    with col_m2:
        st.metric("Área No Cosechada (Pérdida)", f"{perdida_total/1000:.1f}K ha", 
                  delta=f"-{(perdida_total/area_sembrada_total*100):.1f}%", delta_color="inverse")
    with col_m3:
        st.metric("Eficiencia Promedio Nacional", f"{eficiencia_promedio:.1f}%")
    
    st.info("""
    **Interpretación:** La diferencia entre área sembrada y cosechada indica pérdidas por factores climáticos, 
    plagas, enfermedades o problemas logísticos. Una eficiencia menor al 90% puede indicar problemas significativos.
    """)
else:
    st.warning("No hay datos de área cosechada disponibles.")

st.markdown("---")

# ==================== VISUALIZACIÓN 2.2: ANÁLISIS DE RENDIMIENTO ====================
st.markdown("## 2.2 Análisis de Rendimiento y Productividad")

col1, col2 = st.columns(2)

with col1:
    # Scatter plot: Área sembrada vs Producción
    scatter_data = df_filtrado.groupby('departamento').agg({
        'area_sembrada': 'sum',
        'produccion': 'sum',
        'rendimiento': 'mean'
    }).reset_index()
    
    fig_scatter = px.scatter(
        scatter_data,
        x='area_sembrada',
        y='produccion',
        size='rendimiento',
        color='rendimiento',
        hover_name='departamento',
        text='departamento',
        title='Área Sembrada vs Producción por Departamento',
        labels={
            'area_sembrada': 'Área Sembrada (ha)',
            'produccion': 'Producción (ton)',
            'rendimiento': 'Rendimiento (t/ha)'
        },
        color_continuous_scale='Viridis',
        size_max=30
    )
    fig_scatter.update_traces(textposition='top center', textfont=dict(size=8))
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.info("""
    **Interpretación:** Los departamentos más alejados de la línea de tendencia pueden tener 
    rendimientos atípicos (muy altos o muy bajos). El tamaño de la burbuja indica el rendimiento promedio.
    """)

with col2:
    # Box plot de rendimientos por región
    top_10_depts = df_filtrado.groupby('departamento')['produccion'].sum().nlargest(10).index
    df_top10 = df_filtrado[df_filtrado['departamento'].isin(top_10_depts)]
    
    fig_box = px.box(
        df_top10,
        x='departamento',
        y='rendimiento',
        title='Distribución de Rendimientos: Top 10 Departamentos',
        labels={'rendimiento': 'Rendimiento (t/ha)', 'departamento': 'Departamento'},
        color='departamento'
    )
    fig_box.update_layout(height=500, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.info("""
    **Interpretación:** Los box plots muestran la variabilidad del rendimiento. 
    Cajas más estrechas indican consistencia, mientras que outliers señalan casos extremos.
    """)

st.markdown("---")

# ==================== VISUALIZACIÓN 2.3: MATRIZ DE CORRELACIÓN ====================
st.markdown("## 2.3 Correlaciones entre Variables Productivas")

# Preparar datos para correlación
corr_data = df_filtrado[['area_sembrada', 'area_cosechada', 'produccion', 'rendimiento']].copy()
corr_data = corr_data.dropna()

if len(corr_data) > 0:
    correlation_matrix = corr_data.corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Matriz de Correlación entre Variables Productivas',
        labels=dict(color="Correlación"),
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.info("""
    **Interpretación:** Valores cercanos a +1 indican correlación positiva fuerte, 
    cercanos a -1 correlación negativa, y cercanos a 0 ausencia de correlación lineal.
    """)
else:
    st.warning("No hay suficientes datos para calcular correlaciones.")

st.markdown("---")

# ==================== SECCIÓN 3: ANÁLISIS ESPECÍFICO POR CATEGORÍAS ====================
st.markdown("# SECCIÓN 3: Análisis Específico por Categorías")
st.markdown("---")

# ==================== VISUALIZACIÓN 3.1: SISTEMAS PRODUCTIVOS ====================
st.markdown("## 3.1 Sistemas Productivos de Arroz")

if 'sistema_productivo' in df_filtrado.columns and df_filtrado['sistema_productivo'].notna().sum() > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Producción por sistema productivo (ranking completo, mayor arriba)
        sistemas = df_filtrado.groupby('sistema_productivo')['produccion'].sum().sort_values(ascending=True)  # Invertido para gráfico horizontal
        
        fig3 = px.bar(
            x=sistemas.values / 1000,
            y=sistemas.index,
            orientation='h',
            title='Ranking de Sistemas Productivos por Producción',
            labels={'x': 'Producción (K ton)', 'y': 'Sistema Productivo'},
            color=sistemas.values,
            color_continuous_scale='Blues',
            text=[f"{val/1000:.1f}" for val in sistemas.values]
        )
        fig3.update_traces(texttemplate='%{text}K', textposition='outside')
        fig3.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Rendimiento por sistema productivo (ranking completo, mayor a menor)
        rend_sistemas = df_filtrado.groupby('sistema_productivo')['rendimiento'].mean().sort_values(ascending=False)
        
        fig4 = px.bar(
            x=rend_sistemas.index,
            y=rend_sistemas.values,
            title='Ranking de Sistemas por Rendimiento Promedio',
            labels={'x': 'Sistema Productivo', 'y': 'Rendimiento (t/ha)'},
            color=rend_sistemas.values,
            color_continuous_scale='RdYlGn',
            text=rend_sistemas.values
        )
        fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig4.update_layout(height=600, showlegend=False, xaxis_tickangle=-45, xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig4, use_container_width=True)
    
    sistemas_desc = sistemas.sort_values(ascending=False)  # Para estadísticas correctas
    st.markdown(f"""
    ### Análisis de Sistemas Productivos (Específico)
    - **Sistemas identificados:** {df_filtrado['sistema_productivo'].nunique()}
    - **Sistema más productivo:** {sistemas_desc.index[0]}
    - **Mayor rendimiento:** {rend_sistemas.index[0]} ({rend_sistemas.values[0]:.2f} t/ha)
    
    Este análisis específico muestra las diferencias en eficiencia entre sistemas productivos de arroz (riego, secano, mecanizado, etc.).
    """)
else:
    st.info("No hay datos de sistemas productivos disponibles.")

st.markdown("---")

# ==================== VISUALIZACIÓN 3.2: ANÁLISIS ESTACIONAL/PERIÓDICO ====================
st.markdown("## 3.2 Análisis Estacional y por Período")

if 'periodo' in df_filtrado.columns and df_filtrado['periodo'].notna().sum() > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Producción por período
        produccion_periodo = df_filtrado.groupby('periodo')['produccion'].sum().sort_values(ascending=False)
        
        fig_periodo = px.pie(
            values=produccion_periodo.values,
            names=produccion_periodo.index,
            title='Distribución de Producción por Período',
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        fig_periodo.update_traces(textposition='inside', textinfo='percent+label')
        fig_periodo.update_layout(height=500)
        st.plotly_chart(fig_periodo, use_container_width=True)
    
    with col2:
        # Heatmap temporal: Año vs Periodo
        if 'año' in df_filtrado.columns:
            heatmap_data = df_filtrado.groupby(['año', 'periodo'])['produccion'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='periodo', columns='año', values='produccion')
            
            fig_heatmap = px.imshow(
                heatmap_pivot,
                title='Mapa de Calor: Producción por Año y Período',
                labels=dict(x="Año", y="Período", color="Producción (ton)"),
                color_continuous_scale='YlGn',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.info("""
    **Interpretación:** El análisis estacional muestra si hay concentración de producción 
    en ciertos períodos del año, lo que puede indicar patrones climáticos o de siembra.
    """)
else:
    st.info("No hay datos de período disponibles.")

st.markdown("---")

# ==================== SECCIÓN 4: ANÁLISIS ESTADÍSTICO AVANZADO ====================
st.markdown("# SECCIÓN 4: Análisis Estadístico Avanzado")
st.markdown("---")

# ==================== VISUALIZACIÓN 4.1: DISTRIBUCIONES ====================
st.markdown("## 4.1 Distribución de Variables Productivas")

col1, col2 = st.columns(2)

with col1:
    # Histograma de rendimientos
    fig_hist_rend = px.histogram(
        df_filtrado,
        x='rendimiento',
        nbins=30,
        title='Distribución de Rendimientos',
        labels={'rendimiento': 'Rendimiento (t/ha)', 'count': 'Frecuencia'},
        color_discrete_sequence=['green'],
        text_auto=True
    )
    fig_hist_rend.update_layout(
        height=400, 
        showlegend=False,
        xaxis_title='Rendimiento (t/ha)',
        yaxis_title='count'
    )
    st.plotly_chart(fig_hist_rend, use_container_width=True)

with col2:
    # Histograma de producción por municipio
    prod_municipio = df_filtrado.groupby('municipio')['produccion'].sum()
    
    fig_hist_prod = px.histogram(
        x=prod_municipio.values,
        nbins=40,
        title='Distribución de Producción por Municipio',
        labels={'x': 'Producción (ton)', 'count': 'Número de Municipios'},
        color_discrete_sequence=['blue'],
        text_auto=True
    )
    fig_hist_prod.update_layout(
        height=400, 
        showlegend=False,
        xaxis_title='Producción (ton)',
        yaxis_title='count'
    )
    st.plotly_chart(fig_hist_prod, use_container_width=True)

st.info("""
**Interpretación:** Los histogramas muestran la distribución de frecuencias. 
Distribuciones sesgadas pueden indicar concentración en ciertos valores.
""")

st.markdown("---")

# ==================== VISUALIZACIÓN 4.2: COMPARATIVA TOP VS BOTTOM ====================
st.markdown("## 4.2 Comparativa: Top 10 vs Bottom 10 Departamentos")

# Obtener top 10 y bottom 10
produccion_dep_rank = df_filtrado.groupby('departamento')['produccion'].sum().sort_values()

if len(produccion_dep_rank) >= 10:
    bottom_10 = produccion_dep_rank.head(10)
    top_10 = produccion_dep_rank.tail(10)
    
    # Crear DataFrame para gráfico divergente
    comparison_data = pd.DataFrame({
        'Departamento': list(bottom_10.index) + list(top_10.index),
        'Producción': list(-bottom_10.values) + list(top_10.values),
        'Categoría': ['Bottom 10']*10 + ['Top 10']*10
    })
    
    comparison_data['Producción_label'] = comparison_data['Producción'].apply(lambda x: f"{abs(x)/1000:.1f}K" if abs(x) >= 1000 else f"{abs(x):.0f}")
    
    fig_divergente = px.bar(
        comparison_data,
        y='Departamento',
        x='Producción',
        color='Categoría',
        orientation='h',
        title='Comparativa: Top 10 vs Bottom 10 Departamentos por Producción',
        labels={'Producción': 'Producción (ton)', 'Departamento': 'Departamento'},
        color_discrete_map={'Top 10': 'green', 'Bottom 10': 'red'},
        text='Producción_label'
    )
    fig_divergente.update_traces(textposition='outside')
    fig_divergente.update_layout(height=700, xaxis_title='← Bottom 10 | Producción (ton) | Top 10 →')
    st.plotly_chart(fig_divergente, use_container_width=True)
    
    # Estadísticas comparativas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Promedio Top 10", f"{top_10.mean()/1000:.1f}K ton")
    with col2:
        st.metric("Promedio Bottom 10", f"{bottom_10.mean():.0f} ton")
    with col3:
        brecha = (top_10.mean() / bottom_10.mean()) if bottom_10.mean() > 0 else 0
        st.metric("Brecha (veces)", f"{brecha:.1f}x")
    
    st.info("""
    **Interpretación:** Esta visualización divergente contrasta los departamentos más y menos productivos, 
    revelando la magnitud de las diferencias en la producción nacional.
    """)
else:
    st.warning("No hay suficientes departamentos para realizar esta comparación.")

st.markdown("---")

# ==================== VISUALIZACIÓN 4.3: ANÁLISIS DE CONCENTRACIÓN ====================
st.markdown("## 4.3 Análisis de Concentración de la Producción")

# Calcular curva de Lorenz
produccion_municipios_conc = df_filtrado.groupby('municipio')['produccion'].sum().sort_values()
produccion_acum = produccion_municipios_conc.cumsum()
produccion_acum_pct = (produccion_acum / produccion_acum.iloc[-1] * 100)
municipios_pct = np.linspace(0, 100, len(produccion_acum_pct))

# Crear DataFrame para Lorenz
lorenz_data = pd.DataFrame({
    'Municipios (%)': municipios_pct,
    'Producción Acumulada (%)': produccion_acum_pct.values
})

# Agregar línea de igualdad perfecta
lorenz_data_completo = pd.concat([
    pd.DataFrame({'Municipios (%)': [0, 100], 'Producción Acumulada (%)': [0, 100], 'Tipo': ['Igualdad Perfecta', 'Igualdad Perfecta']}),
    lorenz_data.assign(Tipo='Distribución Real')
])

fig_lorenz = px.line(
    lorenz_data_completo,
    x='Municipios (%)',
    y='Producción Acumulada (%)',
    color='Tipo',
    title='Curva de Lorenz: Concentración de la Producción por Municipio',
    labels={'Municipios (%)': 'Porcentaje Acumulado de Municipios', 
            'Producción Acumulada (%)': 'Porcentaje Acumulado de Producción'},
    color_discrete_map={'Distribución Real': 'green', 'Igualdad Perfecta': 'gray'}
)
fig_lorenz.update_traces(line=dict(width=3))
fig_lorenz.update_layout(height=500)
st.plotly_chart(fig_lorenz, use_container_width=True)

# Calcular estadísticas de concentración
top_20_pct_municipios = int(len(produccion_municipios_conc) * 0.2)
produccion_top_20 = produccion_municipios_conc.tail(top_20_pct_municipios).sum()
produccion_total_conc = produccion_municipios_conc.sum()
concentracion_20 = (produccion_top_20 / produccion_total_conc * 100) if produccion_total_conc > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric("Concentración en Top 20% de Municipios", f"{concentracion_20:.1f}%")
with col2:
    st.metric("Total de Municipios Productores", f"{len(produccion_municipios_conc)}")

st.info("""
**Interpretación:** La curva de Lorenz muestra el grado de concentración. Cuanto más alejada esté 
de la línea de igualdad perfecta, mayor es la concentración de la producción en pocos municipios. 
Si el 20% de municipios produce el 80% o más, hay alta concentración.
""")

st.markdown("---")

# ==================== TABLA DE DATOS ====================
st.markdown("## Tabla de Datos Filtrados")

with st.expander("Ver datos detallados (primeras 100 filas)"):
    st.dataframe(
        df_filtrado[['cultivo', 'departamento', 'municipio', 'año', 
                     'area_sembrada', 'area_cosechada', 'produccion', 'rendimiento']].head(100),
        use_container_width=True
    )
    
    # Botón de descarga
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos filtrados (CSV)",
        data=csv,
        file_name=f'cereales_filtrado_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

st.markdown("---")

# ==================== DOCUMENTACIÓN DE DATOS ====================
st.markdown("## Documentación y Fuente de Datos")

with st.expander("Información sobre la fuente de datos y actualización"):
    st.markdown("""
    ### Fuente de Datos
    
    **Origen:** Evaluaciones Agropecuarias Municipales (EVA) - Ministerio de Agricultura y Desarrollo Rural de Colombia

    **Última actualización de datos:** Febrero 2026
    
    **Cultivo analizado:** Arroz
    
    
    ---
    
    ### Descripción del Dataset
    
    - **Registros totales (arroz):** {len(df):,}
    - **Registros filtrados actualmente:** {len(df_filtrado):,}
    - **Variables:** {len(df.columns)}
    - **Período temporal:** {int(df['año'].min()) if 'año' in df.columns else 'N/A'} - {int(df['año'].max()) if 'año' in df.columns else 'N/A'}
    - **Cultivo:** Arroz
    - **Departamentos cubiertos:** {df['departamento'].nunique()}
    - **Municipios cubiertos:** {df['municipio'].nunique()}
    
    ---
    
    ### Información del Desarrollo
    
    **Desarrollado por:** Grupo 2 - MAD (Métodos Analíticos de Datos)
    
    **Fecha de desarrollo:** Febrero 2026
    
    **Herramientas utilizadas:** Python, Streamlit, Plotly, Pandas
    
    **Framework de análisis:** EDA QUEST (Question, Understand, Explore, Scrutinize, Transform)
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Dashboard de Producción de Arroz en Colombia - EDA de Macro a Específico | Grupo 2 MAD | 2026</p>
    <p><small>Desarrollado con Streamlit | Datos: Ministerio de Agricultura de Colombia</small></p>
</div>
""", unsafe_allow_html=True)
