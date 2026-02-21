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
st.title(" Dashboard Analítico: Sepsis en UCI")
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
    st.sidebar.header("Filtros")
    
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
    
    total_patients = len(df_filtered)
    mortality_rate = (df_filtered['outcome'] == 'Fallecido').sum() / total_patients * 100
    avg_age = df_filtered['age'].mean()
    avg_los = df_filtered['icu_length_of_stay_days'].mean()
    avg_sofa = df_filtered['sofa_score'].mean()
    
    # Crear métricas con cajitas sutiles usando HTML/CSS
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 3px solid #1f77b4; padding: 16px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <p style="color: #666; font-size: 13px; margin: 0; font-weight: 500;">Total Pacientes</p>
            <p style="color: #1f77b4; font-size: 28px; font-weight: 600; margin: 8px 0 0 0;">{total_patients:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 3px solid #e74c3c; padding: 16px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <p style="color: #666; font-size: 13px; margin: 0; font-weight: 500;">Tasa de Mortalidad</p>
            <p style="color: #e74c3c; font-size: 28px; font-weight: 600; margin: 8px 0 0 0;">{mortality_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 3px solid #1f77b4; padding: 16px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <p style="color: #666; font-size: 13px; margin: 0; font-weight: 500;">Edad Promedio</p>
            <p style="color: #1f77b4; font-size: 28px; font-weight: 600; margin: 8px 0 0 0;">{avg_age:.1f} años</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 3px solid #1f77b4; padding: 16px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <p style="color: #666; font-size: 13px; margin: 0; font-weight: 500;">Estancia Media UCI</p>
            <p style="color: #1f77b4; font-size: 28px; font-weight: 600; margin: 8px 0 0 0;">{avg_los:.1f} días</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border-left: 3px solid #1f77b4; padding: 16px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
            <p style="color: #666; font-size: 13px; margin: 0; font-weight: 500;">SOFA Score Medio</p>
            <p style="color: #1f77b4; font-size: 28px; font-weight: 600; margin: 8px 0 0 0;">{avg_sofa:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Fila 1: Distribución de severidad y origen de infección
    st.header("Distribución Clínica")
    st.markdown("*Análisis de la distribución de casos por severidad de sepsis y origen anatómico de la infección*")
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
    
    st.divider()
    
    # Fila 2: Series temporales
    st.header("Análisis Temporal")
    st.markdown("*Evolución mensual de ingresos hospitalarios y tasas de mortalidad a lo largo del período de estudio*")
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
    
    st.divider()
    
    # Fila 3: Análisis de mortalidad
    st.header("Análisis de Mortalidad")
    st.markdown("*Comparación de tasas de mortalidad según severidad de sepsis y grupos de edad*")
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
    
    st.divider()
    
    # Fila 4: Intervenciones terapéuticas
    st.header("Intervenciones Terapéuticas")
    st.markdown("*Distribución de pacientes que requirieron soporte avanzado: ventilación mecánica, diálisis y vasopresores*")
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
    
    st.divider()
    
    # Fila 5: Análisis de scores clínicos
    st.header("Scores Clínicos y Parámetros")
    st.markdown("*Distribución de scores de severidad SOFA y APACHE II según el nivel de sepsis y resultado del paciente*")
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
    
    st.divider()
    
    # Fila 6: Estancia en UCI
    st.header("Análisis de Estancia en UCI")
    st.markdown("*Duración de hospitalización en UCI según severidad y resultado clínico del paciente*")
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
    
    st.divider()
    
    # Fila 7: Correlaciones
    st.header("Análisis de Correlaciones")
    st.markdown("*Matriz de correlación entre variables numéricas y explorador interactivo de relaciones entre parámetros clínicos*")
    
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
    st.markdown("*Selecciona variables para visualizar su relación y tendencia*")
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
    
    st.divider()
    
    # Tabla de estadísticas detalladas
    st.header("Estadísticas Detalladas")
    st.markdown("*Estadísticas agregadas por diferentes categorías clínicas*")
    
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
    
    st.divider()
    
    # Detección de anomalías
    st.header("Detección de Anomalías")
    st.markdown("*Identificación de períodos con tasas de mortalidad significativamente superiores al promedio*")
    
    # Filtros para seleccionar mes y año
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        selected_year = st.selectbox(
            "Seleccionar Año",
            options=sorted(df_filtered['year'].unique()),
            index=len(sorted(df_filtered['year'].unique())) - 1  # Último año por defecto
        )
    
    with col_filter2:
        month_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        # Obtener meses disponibles para el año seleccionado
        available_months = sorted(df_filtered[df_filtered['year'] == selected_year]['month'].unique())
        month_options = {month: month_names[month] for month in available_months}
        
        selected_month = st.selectbox(
            "Seleccionar Mes",
            options=list(month_options.keys()),
            format_func=lambda x: month_options[x],
            index=2 if 3 in available_months else 0  # Marzo por defecto si existe
        )
    
    # Analizar el período seleccionado
    selected_period = df_filtered[
        (df_filtered['admission_date'].dt.year == selected_year) & 
        (df_filtered['admission_date'].dt.month == selected_month)
    ]
    
    if len(selected_period) > 0:
        period_mortality = (selected_period['outcome'] == 'Fallecido').sum() / len(selected_period) * 100
        overall_mortality = (df_filtered['outcome'] == 'Fallecido').sum() / len(df_filtered) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Pacientes {month_names[selected_month]} {selected_year}", len(selected_period))
        col2.metric(f"Mortalidad {month_names[selected_month]} {selected_year}", f"{period_mortality:.1f}%", 
                   delta=f"{period_mortality - overall_mortality:.1f}% vs media",
                   delta_color="inverse")
        col3.metric("Mortalidad General", f"{overall_mortality:.1f}%")
        
        if period_mortality > overall_mortality * 1.2:  # 20% mayor que la media
            st.warning(f"ANOMALÍA DETECTADA: La mortalidad en {month_names[selected_month]} {selected_year} ({period_mortality:.1f}%) es significativamente superior a la media general ({overall_mortality:.1f}%). Diferencia de {period_mortality - overall_mortality:.1f} puntos porcentuales.")
        elif period_mortality < overall_mortality * 0.8:  # 20% menor que la media
            st.success(f"PERÍODO FAVORABLE: La mortalidad en {month_names[selected_month]} {selected_year} ({period_mortality:.1f}%) es significativamente inferior a la media general ({overall_mortality:.1f}%). Diferencia de {period_mortality - overall_mortality:.1f} puntos porcentuales.")
    else:
        st.warning(f"No hay datos disponibles para {month_names[selected_month]} {selected_year}.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center'>
        <p>Dashboard Grupo 3 </p>
       <p> Registro de la UCI de un hospital de todos los ingresos por sepsis, días de estancia, la mortalidad y la supervivencia.</p>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Error: No se encontró el archivo 'sepsis_icu_data.csv'. Por favor, asegúrate de que el archivo esté en el mismo directorio que este script.")
    st.info("Ejecuta primero el script de generación de datos o el notebook para crear el archivo CSV.")
except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
    st.exception(e)
