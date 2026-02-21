"""
Dashboard Analítico de Sepsis en UCI
Tablero interactivo con Streamlit para análisis de datos de pacientes con sepsis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Sepsis UCI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏥 Dashboard Analítico: Sepsis en UCI")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('sepsis_icu_data.csv')
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['year_month'] = df['admission_date'].dt.to_period('M').astype(str)
    df['month'] = df['admission_date'].dt.month
    df['year'] = df['admission_date'].dt.year
    return df

try:
    df = load_data()
    
    # Sidebar - Filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de fecha
    min_date = df['admission_date'].min().date()
    max_date = df['admission_date'].max().date()
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtro de severidad
    severities = st.sidebar.multiselect(
        "Severidad de sepsis",
        options=df['sepsis_severity'].unique(),
        default=df['sepsis_severity'].unique()
    )
    
    # Filtro de género
    genders = st.sidebar.multiselect(
        "Género",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
    
    # Filtro de outcome
    outcomes = st.sidebar.multiselect(
        "Resultado",
        options=df['outcome'].unique(),
        default=df['outcome'].unique()
    )
    
    # Aplicar filtros
    if len(date_range) == 2:
        df_filtered = df[
            (df['admission_date'].dt.date >= date_range[0]) &
            (df['admission_date'].dt.date <= date_range[1]) &
            (df['sepsis_severity'].isin(severities)) &
            (df['gender'].isin(genders)) &
            (df['outcome'].isin(outcomes))
        ]
    else:
        df_filtered = df[
            (df['sepsis_severity'].isin(severities)) &
            (df['gender'].isin(genders)) &
            (df['outcome'].isin(outcomes))
        ]
    
    # Métricas principales
    st.header("Métricas Principales")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_patients = len(df_filtered)
    mortality_rate = (df_filtered['outcome'] == 'Fallecido').sum() / total_patients * 100
    avg_age = df_filtered['age'].mean()
    avg_los = df_filtered['icu_length_of_stay_days'].mean()
    avg_sofa = df_filtered['sofa_score'].mean()
    
    col1.metric("Total Pacientes", f"{total_patients:,}")
    col2.metric("Tasa de Mortalidad", f"{mortality_rate:.1f}%")
    col3.metric("Edad Promedio", f"{avg_age:.1f} años")
    col4.metric("Estancia Media UCI", f"{avg_los:.1f} días")
    col5.metric("SOFA Score Medio", f"{avg_sofa:.1f}")
    
    st.markdown("---")
    
    # Fila 1: Distribución de severidad y origen de infección
    st.header("Distribución Clínica")
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de severidad
        severity_counts = df_filtered['sepsis_severity'].value_counts()
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Distribución por Severidad de Sepsis",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig_severity.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Gráfico de origen de infección
        infection_counts = df_filtered['infection_source'].value_counts()
        fig_infection = px.bar(
            x=infection_counts.values,
            y=infection_counts.index,
            orientation='h',
            title="Origen de la Infección",
            labels={'x': 'Número de casos', 'y': 'Origen'},
            color=infection_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_infection.update_layout(showlegend=False)
        st.plotly_chart(fig_infection, use_container_width=True)
    
    # Fila 2: Series temporales
    st.header("Análisis Temporal")
    col1, col2 = st.columns(2)
    
    with col1:
        # Ingresos por mes
        admissions_by_month = df_filtered.groupby('year_month').size().reset_index(name='count')
        fig_admissions = px.line(
            admissions_by_month,
            x='year_month',
            y='count',
            title="Ingresos por Mes",
            labels={'year_month': 'Mes', 'count': 'Número de ingresos'},
            markers=True
        )
        fig_admissions.update_xaxes(tickangle=45)
        st.plotly_chart(fig_admissions, use_container_width=True)
    
    with col2:
        # Mortalidad por mes
        mortality_by_month = df_filtered.groupby('year_month').apply(
            lambda x: (x['outcome'] == 'Fallecido').sum() / len(x) * 100
        ).reset_index(name='mortality_rate')
        
        fig_mortality_trend = px.line(
            mortality_by_month,
            x='year_month',
            y='mortality_rate',
            title="Tasa de Mortalidad Mensual (%)",
            labels={'year_month': 'Mes', 'mortality_rate': 'Mortalidad (%)'},
            markers=True,
            color_discrete_sequence=['red']
        )
        fig_mortality_trend.add_hline(
            y=mortality_rate, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Media: {mortality_rate:.1f}%"
        )
        fig_mortality_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_mortality_trend, use_container_width=True)
    
    # Fila 3: Análisis de mortalidad
    st.header("Análisis de Mortalidad")
    col1, col2 = st.columns(2)
    
    with col1:
        # Mortalidad por severidad
        mortality_by_severity = df_filtered.groupby('sepsis_severity')['outcome'].apply(
            lambda x: (x == 'Fallecido').sum() / len(x) * 100
        ).reset_index(name='mortality_rate')
        
        fig_mort_severity = px.bar(
            mortality_by_severity,
            x='sepsis_severity',
            y='mortality_rate',
            title="Mortalidad por Severidad",
            labels={'sepsis_severity': 'Severidad', 'mortality_rate': 'Mortalidad (%)'},
            color='mortality_rate',
            color_continuous_scale='Reds',
            text='mortality_rate'
        )
        fig_mort_severity.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_mort_severity, use_container_width=True)
    
    with col2:
        # Mortalidad por grupo de edad
        df_filtered['age_group'] = pd.cut(
            df_filtered['age'],
            bins=[0, 40, 50, 60, 70, 80, 100],
            labels=['<40', '40-50', '50-60', '60-70', '70-80', '>80']
        )
        
        mortality_by_age = df_filtered.groupby('age_group')['outcome'].apply(
            lambda x: (x == 'Fallecido').sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index(name='mortality_rate')
        
        fig_mort_age = px.bar(
            mortality_by_age,
            x='age_group',
            y='mortality_rate',
            title="Mortalidad por Grupo de Edad",
            labels={'age_group': 'Grupo de Edad', 'mortality_rate': 'Mortalidad (%)'},
            color='mortality_rate',
            color_continuous_scale='Oranges',
            text='mortality_rate'
        )
        fig_mort_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_mort_age, use_container_width=True)
    
    # Fila 4: Intervenciones terapéuticas
    st.header("Intervenciones Terapéuticas")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ventilation_counts = df_filtered['mechanical_ventilation'].value_counts()
        fig_vent = px.pie(
            values=ventilation_counts.values,
            names=ventilation_counts.index,
            title="Ventilación Mecánica",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        st.plotly_chart(fig_vent, use_container_width=True)
    
    with col2:
        dialysis_counts = df_filtered['dialysis_required'].value_counts()
        fig_dial = px.pie(
            values=dialysis_counts.values,
            names=dialysis_counts.index,
            title="Diálisis Requerida",
            color_discrete_sequence=['#95E1D3', '#F38181']
        )
        st.plotly_chart(fig_dial, use_container_width=True)
    
    with col3:
        vasopressor_counts = df_filtered['vasopressor_use'].value_counts()
        fig_vaso = px.pie(
            values=vasopressor_counts.values,
            names=vasopressor_counts.index,
            title="Uso de Vasopresores",
            color_discrete_sequence=['#AA96DA', '#FCBAD3']
        )
        st.plotly_chart(fig_vaso, use_container_width=True)
    
    # Fila 5: Análisis de scores clínicos
    st.header("Scores Clínicos y Parámetros")
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot de SOFA score por severidad
        fig_sofa = px.box(
            df_filtered,
            x='sepsis_severity',
            y='sofa_score',
            color='outcome',
            title="SOFA Score por Severidad y Resultado",
            labels={'sepsis_severity': 'Severidad', 'sofa_score': 'SOFA Score'},
            color_discrete_map={'Superviviente': '#2ECC71', 'Fallecido': '#E74C3C'}
        )
        st.plotly_chart(fig_sofa, use_container_width=True)
    
    with col2:
        # Box plot de APACHE II score por severidad
        fig_apache = px.box(
            df_filtered,
            x='sepsis_severity',
            y='apache_ii_score',
            color='outcome',
            title="APACHE II Score por Severidad y Resultado",
            labels={'sepsis_severity': 'Severidad', 'apache_ii_score': 'APACHE II Score'},
            color_discrete_map={'Superviviente': '#2ECC71', 'Fallecido': '#E74C3C'}
        )
        st.plotly_chart(fig_apache, use_container_width=True)
    
    # Fila 6: Estancia en UCI
    st.header("Análisis de Estancia en UCI")
    col1, col2 = st.columns(2)
    
    with col1:
        # Estancia por severidad
        fig_los_severity = px.box(
            df_filtered,
            x='sepsis_severity',
            y='icu_length_of_stay_days',
            color='sepsis_severity',
            title="Días de Estancia por Severidad",
            labels={'sepsis_severity': 'Severidad', 'icu_length_of_stay_days': 'Días en UCI'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_los_severity, use_container_width=True)
    
    with col2:
        # Estancia por outcome
        fig_los_outcome = px.violin(
            df_filtered,
            x='outcome',
            y='icu_length_of_stay_days',
            color='outcome',
            title="Distribución de Estancia por Resultado",
            labels={'outcome': 'Resultado', 'icu_length_of_stay_days': 'Días en UCI'},
            color_discrete_map={'Superviviente': '#3498DB', 'Fallecido': '#E74C3C'},
            box=True
        )
        st.plotly_chart(fig_los_outcome, use_container_width=True)
    
    # Fila 7: Correlaciones
    st.header("Análisis de Correlaciones")
    
    # Crear matriz de correlación
    numeric_cols = ['age', 'comorbidities_count', 'sofa_score', 'apache_ii_score',
                   'initial_lactate_mmol', 'wbc_count_thousands', 'temperature_celsius',
                   'systolic_bp_mmhg', 'heart_rate_bpm', 'respiratory_rate_rpm',
                   'icu_length_of_stay_days']
    
    corr_matrix = df_filtered[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Matriz de Correlación de Variables Numéricas",
        labels=dict(color="Correlación"),
        color_continuous_scale='RdBu_r',
        aspect='auto',
        text_auto='.2f'
    )
    fig_corr.update_xaxes(tickangle=45)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter plot interactivo
    st.subheader("Explorador de Relaciones")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_var = st.selectbox("Variable X", numeric_cols, index=2)  # SOFA por defecto
    with col2:
        y_var = st.selectbox("Variable Y", numeric_cols, index=10)  # Estancia por defecto
    with col3:
        color_var = st.selectbox("Color", ['sepsis_severity', 'outcome', 'gender'])
    
    fig_scatter = px.scatter(
        df_filtered,
        x=x_var,
        y=y_var,
        color=color_var,
        title=f"Relación entre {x_var} y {y_var}",
        hover_data=['patient_id', 'age', 'sepsis_severity', 'outcome'],
        opacity=0.6,
        trendline="ols"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tabla de estadísticas detalladas
    st.header("Estadísticas Detalladas")
    
    tab1, tab2, tab3 = st.tabs(["Por Severidad", "Por Origen de Infección", "Por Intervenciones"])
    
    with tab1:
        stats_severity = df_filtered.groupby('sepsis_severity').agg({
            'patient_id': 'count',
            'age': 'mean',
            'sofa_score': 'mean',
            'apache_ii_score': 'mean',
            'icu_length_of_stay_days': 'mean',
            'outcome': lambda x: (x == 'Fallecido').sum() / len(x) * 100
        }).round(2)
        stats_severity.columns = ['Casos', 'Edad Media', 'SOFA Medio', 'APACHE II Medio', 
                                  'Estancia Media (días)', 'Mortalidad (%)']
        st.dataframe(stats_severity, use_container_width=True)
    
    with tab2:
        stats_infection = df_filtered.groupby('infection_source').agg({
            'patient_id': 'count',
            'age': 'mean',
            'sofa_score': 'mean',
            'icu_length_of_stay_days': 'mean',
            'outcome': lambda x: (x == 'Fallecido').sum() / len(x) * 100
        }).round(2)
        stats_infection.columns = ['Casos', 'Edad Media', 'SOFA Medio', 
                                   'Estancia Media (días)', 'Mortalidad (%)']
        st.dataframe(stats_infection.sort_values('Casos', ascending=False), use_container_width=True)
    
    with tab3:
        # Crear resumen de intervenciones
        intervention_summary = pd.DataFrame({
            'Intervención': ['Ventilación Mecánica', 'Diálisis', 'Vasopresores'],
            'Pacientes (Sí)': [
                (df_filtered['mechanical_ventilation'] == 'Sí').sum(),
                (df_filtered['dialysis_required'] == 'Sí').sum(),
                (df_filtered['vasopressor_use'] == 'Sí').sum()
            ],
            'Pacientes (No)': [
                (df_filtered['mechanical_ventilation'] == 'No').sum(),
                (df_filtered['dialysis_required'] == 'No').sum(),
                (df_filtered['vasopressor_use'] == 'No').sum()
            ]
        })
        intervention_summary['% Sí'] = (intervention_summary['Pacientes (Sí)'] / 
                                        (intervention_summary['Pacientes (Sí)'] + 
                                         intervention_summary['Pacientes (No)']) * 100).round(2)
        st.dataframe(intervention_summary, use_container_width=True)
    
    # Detección de anomalías
    st.header("Detección de Anomalías")
    
    # Analizar marzo 2025
    march_2025 = df_filtered[
        (df_filtered['admission_date'].dt.year == 2025) & 
        (df_filtered['admission_date'].dt.month == 3)
    ]
    
    if len(march_2025) > 0:
        march_mortality = (march_2025['outcome'] == 'Fallecido').sum() / len(march_2025) * 100
        overall_mortality = (df_filtered['outcome'] == 'Fallecido').sum() / len(df_filtered) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Pacientes Marzo 2025", len(march_2025))
        col2.metric("Mortalidad Marzo 2025", f"{march_mortality:.1f}%", 
                   delta=f"{march_mortality - overall_mortality:.1f}% vs media",
                   delta_color="inverse")
        col3.metric("Mortalidad General", f"{overall_mortality:.1f}%")
        
        if march_mortality > overall_mortality * 1.2:  # 20% mayor que la media
            st.warning(f"⚠️ ANOMALÍA DETECTADA: La mortalidad en Marzo 2025 ({march_mortality:.1f}%) es significativamente superior a la media general ({overall_mortality:.1f}%). Diferencia de {march_mortality - overall_mortality:.1f} puntos porcentuales.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center'>
        <p>Dashboard generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total de registros: {len(df_filtered):,} | Período: {df_filtered['admission_date'].min().strftime('%Y-%m-%d')} a {df_filtered['admission_date'].max().strftime('%Y-%m-%d')}</p>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Error: No se encontró el archivo 'sepsis_icu_data.csv'. Por favor, asegúrate de que el archivo esté en el mismo directorio que este script.")
    st.info("💡 Ejecuta primero el script de generación de datos o el notebook para crear el archivo CSV.")
except Exception as e:
    st.error(f"❌ Error al cargar los datos: {str(e)}")
    st.exception(e)
