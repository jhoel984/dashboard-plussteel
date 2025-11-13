import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard PlusSteel",
    # Usa el logo circular rojo para el favicon
    page_icon="assets/logo2.png",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTES Y RUTAS
# =============================================================================
BASE_EXPORT_DIR = 'models_export'
DATA_FILE = '08_series_temporales_clase_a_con_kmeans.xlsx'
DIR_BASE_MODELOS = os.path.join(BASE_EXPORT_DIR, 'doble_clasificador')
DIR_REENTRENADOS = os.path.join(BASE_EXPORT_DIR, 'reentrenamiento_doble_clasif')
DIR_UMBRALES = os.path.join(BASE_EXPORT_DIR, 'optimizacion_umbral')
PLANIFICACION_FILE = 'planificacion_demanda.csv'

# Archivos de modelos para monitorear fechas
MODELOS_BASE_FILE = os.path.join(DIR_BASE_MODELOS, 'modelos_doble_clasif.pkl')
MODELOS_REENTRENADOS_FILE = os.path.join(DIR_REENTRENADOS, 'modelos_reentrenados.pkl')


# =============================================================================
# FUNCIONES AUXILIARES (DE TU NOTEBOOK)
# =============================================================================
# (Las funciones preparar_datos_doble_clasificador y predecir_producto
# se mantienen exactamente igual que en tu script original)
def preparar_datos_doble_clasificador(df_segmento):
    """
    Prepara datos para modelo doble-clasificador (desde Celda 10.5)
    """
    from sklearn.preprocessing import StandardScaler
    from numpy import log1p
    
    df_prep = df_segmento.copy()
    df_prep['fecha'] = pd.to_datetime(df_prep['fecha'])
    df_prep['venta_ocurrio'] = (df_prep['cantidad_vendida'] > 0).astype(int)
    
    ventas_positivas = df_prep[df_prep['venta_ocurrio'] == 1]['cantidad_vendida']
    limites_rangos = None
    df_prep['rango_venta'] = -1
    
    if len(ventas_positivas) > 10:
        q_50 = ventas_positivas.quantile(0.50)
        limites_rangos = {'q_50': float(q_50)}
        indices_venta = df_prep[df_prep['venta_ocurrio'] == 1].index
        df_prep.loc[indices_venta, 'rango_venta'] = pd.cut(
            ventas_positivas,
            bins=[-np.inf, q_50, np.inf],
            labels=[0, 1],
            right=False
        ).astype(int)
    
    features_list = []
    for producto in df_prep['producto'].unique():
        df_prod = df_prep[df_prep['producto'] == producto].sort_values('fecha').copy()
        
        cantidad_segura = df_prod['cantidad_vendida'].replace(0, np.nan)
        df_prod['precio_implicito'] = df_prod['venta_bs'] / cantidad_segura
        df_prod['precio_implicito'] = df_prod['precio_implicito'].ffill().bfill().fillna(0)
        
        precio_medio_hist = df_prod['precio_implicito'][df_prod['precio_implicito'] > 0].mean()
        if precio_medio_hist > 0:
            df_prod['precio_relativo'] = (df_prod['precio_implicito'] / precio_medio_hist) - 1.0
        else:
            df_prod['precio_relativo'] = 0.0
        
        df_prod['hubo_descuento_implicito'] = (df_prod['precio_relativo'] < -0.05).astype(int)
        df_prod['fecha_ultima_compra'] = df_prod['fecha'].where(df_prod['venta_ocurrio'] == 1).ffill()
        
        if 'fecha_ultima_compra' in df_prod.columns and not df_prod['fecha_ultima_compra'].isna().all():
            df_prod['dias_desde_ultima_compra'] = (df_prod['fecha'] - df_prod['fecha_ultima_compra']).dt.days
            df_prod['dias_desde_ultima_compra'] = df_prod['dias_desde_ultima_compra'].fillna(0).astype(int)
        else:
            df_prod['dias_desde_ultima_compra'] = 0
        
        racha_no_venta = (df_prod['venta_ocurrio'] != 0).cumsum()
        df_prod['racha_de_no_venta'] = df_prod.groupby(racha_no_venta).cumcount()
        
        df_prod['mes'] = df_prod['fecha'].dt.month
        df_prod['trimestre'] = df_prod['fecha'].dt.quarter
        df_prod['mes_sin'] = np.sin(2 * np.pi * df_prod['mes'] / 12)
        df_prod['mes_cos'] = np.cos(2 * np.pi * df_prod['mes'] / 12)
        
        for lag in [1, 3, 6]:
            df_prod[f'lag_{lag}'] = df_prod['cantidad_vendida'].shift(lag)
            df_prod[f'venta_ocurrio_lag_{lag}'] = df_prod['venta_ocurrio'].shift(lag)
        
        for ventana in [3, 6]:
            df_prod[f'media_movil_{ventana}'] = (
                df_prod['cantidad_vendida'].shift(1)
                .rolling(window=ventana, min_periods=1).mean()
            )
        
        for ventana in [3, 6, 12]:
            df_prod[f'tasa_ocurrencia_{ventana}'] = (
                df_prod['venta_ocurrio'].shift(1)
                .rolling(window=ventana, min_periods=1).mean()
            )
        
        features_list.append(df_prod)
    
    if not features_list:
        return pd.DataFrame(), [], None
    
    df_features = pd.concat(features_list, ignore_index=True)
    feature_cols = [col for col in df_features.columns if col not in
                        ['producto', 'fecha', 'cantidad_vendida', 'venta_bs',
                        'num_transacciones', 'periodo', 'venta_ocurrio',
                        'subclase_kmeans', 'grupo_venta', 'rango_venta',
                        'fecha_ultima_compra', 'precio_implicito']]
    
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_features, feature_cols, limites_rangos

def predecir_producto(df_single_row, segmento, modelos, scalers, features, limites, umbrales):
    """
    Predice probabilidades para una fila (desde Celda 12)
    """
    try:
        modelo_dict = modelos[segmento]
        scaler = scalers[segmento]
        feature_cols = features[segmento]
        umbral = umbrales.get(segmento, 0.5)
        limite = limites[segmento]
    except KeyError as e:
        return None
    
    X_pred = df_single_row.reindex(columns=feature_cols).fillna(0)
    X_scaled = scaler.transform(X_pred)
    
    cls_binario = modelo_dict.get('clasificador_binario')
    cls_rangos = modelo_dict.get('clasificador_rangos')
    
    if cls_rangos is None:
        return None
    
    if cls_binario is None:
        prob_venta = 1.0
    else:
        prob_venta = cls_binario.predict_proba(X_scaled)[0, 1]
    
    prob_rangos = cls_rangos.predict_proba(X_scaled)[0]
    prob_no_venta = 1.0 - prob_venta
    prob_venta_baja = prob_venta * prob_rangos[0]
    prob_venta_alta = prob_venta * prob_rangos[1]
    
    return {
        'segmento': segmento,
        'limite_baja_alta': limite.get('q_50', 0), 
        'P(No Venta)': prob_no_venta,
        'P(Venta Baja)': prob_venta_baja,
        'P(Venta Alta)': prob_venta_alta,
        'prob_total_venta': prob_venta,
        'umbral_optimizado_usado': umbral 
    }

# =============================================================================
# CARGA DE DATOS (CACHEADA Y EN SESSION STATE)
# =============================================================================

@st.cache_data
def cargar_artefactos():
    """
    Carga modelos, scalers, datos históricos y recomendaciones
    """
    try:
        # Modelos base
        modelos_base = joblib.load(MODELOS_BASE_FILE)
        scalers_prod = joblib.load(os.path.join(DIR_BASE_MODELOS, 'scalers_doble_clasif.pkl'))
        features_prod = joblib.load(os.path.join(DIR_BASE_MODELOS, 'features_doble_clasif.pkl'))
        limites_prod = joblib.load(os.path.join(DIR_BASE_MODELOS, 'limites_rangos.pkl'))
        
        # Umbrales
        try:
            with open(os.path.join(DIR_UMBRALES, 'umbrales_optimizados.json'), 'r') as f:
                umbrales_prod = json.load(f)
        except FileNotFoundError:
            umbrales_prod = {}
        
        # Modelos reentrenados
        try:
            modelos_reentrenados = joblib.load(MODELOS_REENTRENADOS_FILE)
            scalers_reentrenados = joblib.load(os.path.join(DIR_REENTRENADOS, 'scalers_reentrenados.pkl'))
        except FileNotFoundError:
            modelos_reentrenados = {}
            scalers_reentrenados = {}
        
        # Consolidar
        modelos_prod = modelos_base.copy()
        scalers_final = scalers_prod.copy()
        for seg, modelo in modelos_reentrenados.items():
            if seg in modelos_prod:
                modelos_prod[seg] = modelo
                scalers_final[seg] = scalers_reentrenados[seg]
        
        # Datos históricos
        df_series = pd.read_excel(DATA_FILE)
        
        # Recomendaciones (AHORA PLANIFICACION)
        try:
            df_planif = pd.read_csv(PLANIFICACION_FILE)
        except FileNotFoundError:
            df_planif = None
        
        return True, modelos_prod, scalers_final, features_prod, limites_prod, umbrales_prod, df_series, df_planif
    
    except Exception as e:
        st.error(f"Error cargando artefactos: {e}")
        return False, None, None, None, None, None, None, None

# Lógica de carga con st.session_state
if 'datos_cargados' not in st.session_state:
    with st.spinner("Cargando modelos y datos..."):
        (
            success,
            st.session_state['modelos_prod'],
            st.session_state['scalers_prod'],
            st.session_state['features_prod'],
            st.session_state['limites_prod'],
            st.session_state['umbrales_prod'],
            st.session_state['df_series'],
            st.session_state['df_planif']
        ) = cargar_artefactos()
        st.session_state['datos_cargados'] = success

if not st.session_state['datos_cargados']:
    st.error("No se pudieron cargar los artefactos. Verifica que existan los archivos de modelo.")
    st.stop()
else:
    # MODIFICADO: Emoji eliminado
    st.toast("Modelos y datos cargados.")


# =============================================================================
# SIDEBAR - NAVEGACIÓN Y DATOS
# =============================================================================
with st.sidebar:
    # Usa el logo circular rojo
    st.image("assets/logo1.png", use_container_width=True) 
    
    # MODIFICADO: Menú de navegación sin emojis
    st.markdown("## Menú Principal")
    pagina_seleccionada = st.radio(
        "Seleccione una página:",
        [
            "Inicio", 
            "Pronóstico", 
            "Planificación de Demanda", 
            "Segmentación K-Means", 
            "Salud del Modelo (MLOps)"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Información del Sistema")
    
    # Acceder a los datos desde session_state
    df_series = st.session_state['df_series']
    modelos_prod = st.session_state['modelos_prod']
    
    st.markdown(f"""
    - **Productos Clase A:** {len(df_series['producto'].unique())}
    - **Segmentos K-Means:** {df_series['subclase_kmeans'].nunique()}
    - **Modelos Cargados:** {len(modelos_prod)}
    - **Rango de Datos:** {pd.to_datetime(df_series['fecha']).min().strftime('%Y-%m')} - {pd.to_datetime(df_series['fecha']).max().strftime('%Y-%m')}
    """)
    
    # Fecha de actualización de modelos
    try:
        if os.path.exists(MODELOS_REENTRENADOS_FILE):
            fecha_mod = os.path.getmtime(MODELOS_REENTRENADOS_FILE)
        else:
            fecha_mod = os.path.getmtime(MODELOS_BASE_FILE)
        fecha_str = datetime.fromtimestamp(fecha_mod).strftime('%Y-%m-%d %H:%M')
        st.info(f"Última act. de modelos:\n**{fecha_str}**")
    except Exception as e:
        st.warning(f"No se pudo leer la fecha del modelo.")
    
    st.markdown("---")
    st.markdown("### Tecnologías Usadas")
    st.markdown("""
    - Streamlit
    - Scikit-learn
    - Random Forest
    - K-Means Clustering
    - Plotly
    """)


# =============================================================================
# INTERFAZ PRINCIPAL - CONTENIDO CONDICIONAL
# =============================================================================

# =============================================================================
# PÁGINA 1: INICIO (NUEVA)
# =============================================================================
if pagina_seleccionada == "Inicio":
    
    st.title("Bienvenido al Dashboard de PlusSteel")
    st.markdown("Sistema de Pronóstico de Demanda con Inteligencia Artificial")
    
    

    st.subheader("Quiénes Somos")
    st.markdown("""
    Somos una empresa que nació con el firme compromiso de ser líderes en la fabricación de perfilería metálica
    y sistemas para la construcción en seco **DRY WALL** y **STEEL FRAME**®. 
    
    Nuestra certificación **ISO 9001** válida nuestro compromiso con la excelencia y la satisfacción del cliente.
    """)
    
    st.markdown("---")

    st.subheader("Nuestros Valores")
    col1, col2, col3 = st.columns(3)
    with col1:
        # MODIFICADO: Emoji eliminado
        st.markdown("#### Calidad ISO 9001")
        st.markdown("Somos la Primera Fábrica Boliviana en la fabricación de perfiles de acero galvanizado para construcción en seco con certificación ISO 9001.")
    with col2:
        # MODIFICADO: Emoji eliminado
        st.markdown("#### Experiencia")
        st.markdown("Vasta experiencia en cerchas metálicas, cielos falsos, muros divisorios y fachadas en proyectos públicos y privados.")
    with col3:
        # MODIFICADO: Emoji eliminado
        st.markdown("#### Asesoramiento Profesional")
        st.markdown("Profundo conocimiento de nuestros productos y servicios para ayudarte a tomar decisiones informadas que impulsen el éxito de tu proyecto.")

    st.markdown("---")

    st.subheader("Categorías Principales de Productos")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Perfilería:** Perfiles estructurales STEEL FRAME y perfiles livianos DRY WALL.
        - **Construcción:** Aislamientos térmicos/acústicos, Cercha metálica, Cielo falso.
        - **Complementos:** Variedad en cintas, mallas y tornillos.
        - **Placas:** Amplia gama en placas cementicias, de yeso y desmontables.
        """)
    with col2:
        st.markdown("""
        - **Paneles para Fachadas:** Paneles de Aluminio Compuesto, Hunter Douglas y Lamitech.
        - **Revestimiento Interiores:** Hunter Douglas Interior y Pertech by Lamitech.
        - **Herramientas**
        - **Protección Personal**
        """)

# =============================================================================
# PÁGINA 2: PRONÓSTICO (Antes Tab 1)
# =============================================================================
elif pagina_seleccionada == "Pronóstico":

    st.header("Pronóstico de Demanda por Producto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        lista_productos = sorted(st.session_state['df_series']['producto'].unique())
        producto_sel = st.selectbox(
            "Seleccione un producto Clase A:",
            lista_productos,
            key="producto_pronostico"
        )
    
    with col2:
        st.metric("Total de productos Clase A", len(lista_productos))
    
    # MODIFICADO: Emoji eliminado
    if st.button("Generar Pronóstico", type="primary"):
        with st.spinner(f"Generando pronóstico para '{producto_sel}'..."):
            try:
                df_historial = st.session_state['df_series'][st.session_state['df_series']['producto'] == producto_sel].copy()
                df_historial['fecha'] = pd.to_datetime(df_historial['fecha'])
                segmento = df_historial['subclase_kmeans'].iloc[0]
                
                # Validar modelo
                if segmento not in st.session_state['modelos_prod']:
                    # MODIFICADO: Emoji eliminado
                    st.error(f"No existe modelo entrenado para el segmento '{segmento}'")
                    st.stop()
                
                # Preparar predicción
                fecha_ultima = df_historial['fecha'].max()
                fecha_pred = (fecha_ultima + relativedelta(months=1)).replace(day=1)
                
                df_futuro = pd.DataFrame([{
                    'producto': producto_sel,
                    'fecha': fecha_pred,
                    'cantidad_vendida': 0,
                    'venta_bs': 0,
                    'num_transacciones': 0,
                    'subclase_kmeans': segmento
                }])
                
                df_combined = pd.concat([df_historial, df_futuro], ignore_index=True)
                df_features, feature_cols, _ = preparar_datos_doble_clasificador(df_combined)
                
                if df_features.empty:
                    st.error("No se pudieron generar features para el pronóstico.")
                    st.stop()

                df_pred = df_features.iloc[[-1]]
                
                resultado = predecir_producto(
                    df_pred, segmento, 
                    st.session_state['modelos_prod'], 
                    st.session_state['scalers_prod'],
                    st.session_state['features_prod'], 
                    st.session_state['limites_prod'], 
                    st.session_state['umbrales_prod']
                )
                
                if resultado:
                    # Mostrar resultados
                    st.subheader(f"Pronóstico para {fecha_pred.strftime('%B %Y')}")
                    st.info(f"Producto: **{producto_sel}** | Segmento: **{segmento}**")
                    
                    limite = resultado['limite_baja_alta']
                    
                    # MODIFICADO: Emojis eliminados de recomendaciones
                    if resultado['P(No Venta)'] > 0.6:
                        st.warning(f"**Recomendación: Stock Mínimo.**\n\nAlta probabilidad ({resultado['P(No Venta)']:.1%}) de no vender.")
                    elif resultado['P(Venta Alta)'] > resultado['P(Venta Baja)'] and resultado['P(Venta Alta)'] > 0.35:
                        st.success(f"**Recomendación: Priorizar Reabastecimiento.**\n\nAlta demanda esperada (Prob. Venta Alta: {resultado['P(Venta Alta)']:.1%}).")
                    else:
                        st.info(f"**Recomendación: Demanda Moderada.**\n\nReabastecer con cautela (Prob. Venta Baja: {resultado['P(Venta Baja)']:.1%}).")

                    st.markdown("---")
                    
                    # Gráfico de Donut para probabilidades
                    col_chart, col_hist = st.columns([1, 2])
                    
                    with col_chart:
                        st.markdown(f"**Probabilidades (Límite: {limite:.0f}u)**")
                        
                        labels = [f"No Venta", f"Venta Baja (<{limite:.0f}u)", f"Venta Alta (>{limite:.0f}u)"]
                        values = [resultado['P(No Venta)'], resultado['P(Venta Baja)'], resultado['P(Venta Alta)']]
                        
                        # Colores (Rojo acento, Amarillo advertencia, Gris neutral)
                        colors = ['#FFC107', '#B0B0B0', '#CC0000']

                        fig_pie = go.Figure(data=[go.Pie(
                            labels=labels, 
                            values=values, 
                            hole=.5,
                            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=1)),
                            textinfo='percent+label',
                            textfont_size=14,
                            textfont_color="#FFFFFF"
                        )])
                        fig_pie.update_layout(
                            title_text="Distribución de Probabilidad",
                            title_x=0.5,
                            height=400,
                            showlegend=False,
                            margin=dict(t=50, b=0, l=0, r=0),
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color="#FFFFFF"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_hist:
                        # Gráfico histórico
                        st.markdown("**Historial de Ventas**")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_historial['fecha'],
                            y=df_historial['cantidad_vendida'],
                            mode='lines+markers',
                            name='Histórico',
                            # El gráfico usará el primaryColor (Plata) por defecto
                            line=dict(width=2) 
                        ))
                        
                        fig.add_vline(
                            x=fecha_pred.timestamp() * 1000, 
                            line_width=2, 
                            line_dash="dash", 
                            line_color="white", # Línea blanca
                            annotation_text="Pronóstico",
                            annotation_position="top left"
                        )
                        
                        fig.update_layout(
                            title=f"Serie Temporal: {producto_sel}",
                            xaxis_title="Fecha",
                            yaxis_title="Cantidad Vendida",
                            hovermode='x unified',
                            height=400,
                            margin=dict(t=50, b=0, l=0, r=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color="#FFFFFF"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # MODIFICADO: Emoji eliminado
                    with st.expander("Ver JSON Detallado del Pronóstico"):
                        st.json(resultado)
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# =============================================================================
# PÁGINA 3: PLANIFICACIÓN (Antes Tab 2)
# =============================================================================
elif pagina_seleccionada == "Planificación de Demanda":

    # Función para colorear la tabla (Tema Oscuro)
    def color_accion(accion):
        if "PRIORIZAR" in accion:
            # Texto rojo claro, fondo oscuro
            color = 'color: #FF8A8A; font-weight: bold;' 
        elif "STOCK MÍNIMO" in accion:
            # Texto amarillo, fondo oscuro
            color = 'color: #FFF59D; font-weight: bold;' 
        elif "MODERADO" in accion:
            # Texto verde claro, fondo oscuro
            color = 'color: #A5D6A7;'
        else:
            color = 'color: #FFFFFF' # Texto blanco por defecto
        return color

    st.header("Planificación de Demanda y Stock de Seguridad")
    
    df_planif = st.session_state['df_planif']
    
    if df_planif is not None:
        st.markdown(f"**Total de productos Clase A:** {len(df_planif)}")
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            filtro_accion = st.multiselect(
                "Filtrar por Acción:",
                options=df_planif['accion_recomendada'].unique() if 'accion_recomendada' in df_planif.columns else [],
                default=None
            )
        with col2:
            filtro_segmento = st.multiselect(
                "Filtrar por Segmento:",
                options=df_planif['segmento'].unique() if 'segmento' in df_planif.columns else [],
                default=None
            )
        
        # Aplicar filtros
        df_filtrado = df_planif.copy()
        if filtro_accion:
            df_filtrado = df_filtrado[df_filtrado['accion_recomendada'].isin(filtro_accion)]
        if filtro_segmento:
            df_filtrado = df_filtrado[df_filtrado['segmento'].isin(filtro_segmento)]
        
        # Columnas a mostrar
        columnas_visibles = [
            'producto', 
            'segmento', 
            'accion_recomendada', 
            'stock_seguridad_sugerido',
            'demanda_esperada',
            'punto_reorden',
            'prob_venta_alta',
            'prob_no_venta',
            'venta_promedio_6m'
        ]
        
        # Aplicar estilo de color a la tabla
        st.dataframe(
            df_filtrado[columnas_visibles].style
            .set_properties(**{'color': '#FFFFFF'})  # Texto blanco
            .applymap(
                color_accion, subset=['accion_recomendada']
            ).format({
                'stock_seguridad_sugerido': '{:.1f}',
                'demanda_esperada': '{:.1f}',
                'punto_reorden': '{:.1f}',
                'prob_venta_alta': '{:.1%}',
                'prob_no_venta': '{:.1%}',
                'venta_promedio_6m': '{:.1f}'
            }),
            use_container_width=True,
            height=500
        )
        
        # Resumen
        st.subheader("Resumen de Acciones")
        if 'accion_recomendada' in df_filtrado.columns:
            resumen = df_filtrado['accion_recomendada'].value_counts()
            
            # Colores del pie chart para tema oscuro
            color_map = {
                "PRIORIZAR REABASTECIMIENTO": "#CC0000",
                "STOCK MÍNIMO (ALERTA)": "#FFC107",
                "REABASTECIMIENTO MODERADO": "#A5D6A7"
            }
            nombres_resumen = resumen.index.tolist()
            colores_mapeados = [color_map.get(nombre, '#B0B0B0') for nombre in nombres_resumen]

            fig = px.pie(
                values=resumen.values,
                names=nombres_resumen,
                title="Distribución de Acciones Recomendadas",
            )
            fig.update_traces(
                marker=dict(colors=colores_mapeados, line=dict(color='#FFFFFF', width=1)),
                textfont_color="#000000"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#FFFFFF"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        # MODIFICADO: Emoji eliminado
        st.warning(f"No se encontró el archivo '{PLANIFICACION_FILE}'.")
        st.info("Ejecuta la **Celda 13** del notebook para generar el reporte de planificación.")

# =============================================================================
# PÁGINA 4: SEGMENTACIÓN (Antes Tab 3)
# =============================================================================
elif pagina_seleccionada == "Segmentación K-Means":

    st.header("Visualización de Segmentación K-Means")
    
    # Preparar datos para scatter plot
    df_agregado = st.session_state['df_series'].groupby(['producto', 'subclase_kmeans']).agg({
        'cantidad_vendida': 'sum',
        'venta_bs': 'sum',
        'fecha': 'count'
    }).reset_index()
    df_agregado.columns = ['producto', 'segmento', 'qty_total', 'venta_total', 'frecuencia']
    
    # Aplicar log
    df_agregado['qty_log'] = np.log1p(df_agregado['qty_total'])
    df_agregado['venta_log'] = np.log1p(df_agregado['venta_total'])
    df_agregado['freq_log'] = np.log1p(df_agregado['frecuencia'])
    
    # Scatter plot interactivo
    fig = px.scatter(
        df_agregado,
        x='venta_log',
        y='freq_log',
        color='segmento',
        size='qty_log',
        hover_data=['producto', 'qty_total', 'venta_total', 'frecuencia'],
        title="Segmentación K-Means de Productos Clase A",
        labels={
            'venta_log': 'Venta Total (log)',
            'freq_log': 'Frecuencia de Ventas (log)',
            'segmento': 'Segmento'
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#FFFFFF"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas por segmento
    st.subheader("Estadísticas por Segmento")
    stats = df_agregado.groupby('segmento').agg({
        'producto': 'count',
        'qty_total': ['mean', 'sum'],
        'venta_total': ['mean', 'sum'],
        'frecuencia': 'mean'
    }).round(2)
    stats.columns = ['_'.join(col) for col in stats.columns.values] # Aplanar multi-index
    stats = stats.rename(columns={
        'producto_count': 'N° Productos',
        'qty_total_mean': 'Qty Promedio',
        'qty_total_sum': 'Qty Total',
        'venta_total_mean': 'Venta Promedio',
        'venta_total_sum': 'Venta Total',
        'frecuencia_mean': 'Frecuencia Promedio'
    })
    st.dataframe(
        stats.style.set_properties(**{'color': '#FFFFFF'}), # Texto blanco
        use_container_width=True
    )

# =============================================================================
# PÁGINA 5: SALUD DEL MODELO (Antes Tab 4)
# =============================================================================
elif pagina_seleccionada == "Salud del Modelo (MLOps)":

    st.header("Monitoreo de Salud del Modelo - Detección de Drift")
    
    st.markdown("""
    Esta sección muestra el estado de **drift** (deriva) del modelo, 
    evaluando el rendimiento actual vs. baseline en datos recientes.
    """)
    
    # Configuración de umbrales
    UMBRAL_CRITICO_F1_BINARIO = 0.85
    UMBRAL_CRITICO_AUC_RANGOS = 0.55
    UMBRAL_DRIFT_F1_BINARIO = 0.95
    UMBRAL_DRIFT_AUC_RANGOS = 0.90
    
    # Intentar cargar reporte de monitoreo existente
    reporte_path = os.path.join(DIR_REENTRENADOS, 'reporte_monitoreo.csv')
    alertas_path = os.path.join(DIR_REENTRENADOS, 'alertas_mlops.csv')
    
    try:
        df_reporte = pd.read_csv(reporte_path)
        
        # Resumen general
        col1, col2, col3 = st.columns(3)
        total_modelos = len(df_reporte)
        modelos_ok = (df_reporte['reentrenado'] == False).sum()
        modelos_reentrenados = df_reporte['reentrenado'].sum()
        
        col1.metric("Total de Modelos", total_modelos)
        col2.metric("🟢 Modelos Estables", modelos_ok)
        col3.metric("🔴 Modelos Re-entrenados", int(modelos_reentrenados))
        
        st.markdown("---")
        
        # Análisis detallado por segmento
        for idx, row in df_reporte.iterrows():
            segmento = row['segmento']
            f1_baseline = row['f1_baseline']
            f1_actual = row['f1_actual']
            auc_baseline = row['auc_baseline']
            auc_actual = row['auc_actual']
            reentrenado = row['reentrenado']
            
            # MODIFICADO: Emoji eliminado
            with st.expander(f"Segmento: {segmento}", expanded=(reentrenado == True)):
                # Determinar estado del modelo
                alertas = []
                
                # Chequeos críticos
                if pd.notna(f1_actual) and f1_actual < UMBRAL_CRITICO_F1_BINARIO:
                    alertas.append(('CRITICAL', f'F1 Binario críticamente bajo: {f1_actual:.3f} < {UMBRAL_CRITICO_F1_BINARIO}'))
                
                if pd.notna(auc_actual) and auc_actual < UMBRAL_CRITICO_AUC_RANGOS:
                    alertas.append(('CRITICAL', f'AUC Rangos críticamente bajo: {auc_actual:.3f} < {UMBRAL_CRITICO_AUC_RANGOS}'))
                
                # Chequeos de drift
                if pd.notna(f1_actual) and pd.notna(f1_baseline) and f1_actual < f1_baseline * UMBRAL_DRIFT_F1_BINARIO:
                    alertas.append(('WARNING', f'Drift en F1 Binario: {f1_actual:.3f} < {f1_baseline * UMBRAL_DRIFT_F1_BINARIO:.3f}'))
                
                if pd.notna(auc_actual) and pd.notna(auc_baseline) and auc_actual < auc_baseline * UMBRAL_DRIFT_AUC_RANGOS:
                    alertas.append(('WARNING', f'Drift en AUC Rangos: {auc_actual:.3f} < {auc_baseline * UMBRAL_DRIFT_AUC_RANGOS:.3f}'))
                
                # Determinar estado general
                if reentrenado:
                    estado = "🔴 MODELO RE-ENTRENADO"
                    color = "red"
                    mensaje = "El modelo fue re-entrenado debido a degradación crítica del rendimiento."
                elif any(a[0] == 'CRITICAL' for a in alertas):
                    estado = "🔴 CRÍTICO"
                    color = "red"
                    mensaje = "Rendimiento crítico detectado. Re-entrenamiento necesario."
                elif any(a[0] == 'WARNING' for a in alertas):
                    estado = "🟡 ADVERTENCIA"
                    color = "orange"
                    mensaje = "Se detectó drift moderado. Monitoreo recomendado."
                else:
                    estado = "🟢 OK"
                    color = "green"
                    mensaje = "El modelo está estable y funcionando correctamente."
                
                # Mostrar estado
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"### {estado}")
                with col2:
                    # Calcular score de salud (0-1)
                    if pd.notna(f1_actual):
                        health_score = min(f1_actual / f1_baseline if f1_baseline > 0 else 1.0, 1.0)
                    else:
                        health_score = 0.5 # Default si no hay datos
                    st.progress(health_score)
                    st.caption(f"Score de Salud: {health_score:.1%}")
                
                st.markdown(f":{color}[{mensaje}]")
                
                # Mostrar alertas específicas
                if alertas:
                    # MODIFICADO: Emoji eliminado
                    st.markdown("**Alertas detectadas:**")
                    for tipo, msg in alertas:
                        if tipo == 'CRITICAL':
                            # MODIFICADO: Emoji eliminado
                            st.error(f"{msg}")
                        else:
                            # MODIFICADO: Emoji eliminado
                            st.warning(f"{msg}")
                
                # Métricas comparativas
                st.markdown("**📊 Métricas de Rendimiento:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Precisión Ocurrencia (F1)",
                        f"{f1_actual:.3f}" if pd.notna(f1_actual) else "N/A",
                        delta=f"{f1_actual - f1_baseline:.3f}" if pd.notna(f1_actual) and pd.notna(f1_baseline) else None
                    )
                
                with col2:
                    st.metric(
                        "F1 Baseline",
                        f"{f1_baseline:.3f}" if pd.notna(f1_baseline) else "N/A"
                    )
                
                with col3:
                    st.metric(
                        "Precisión Volumen (AUC)",
                        f"{auc_actual:.3f}" if pd.notna(auc_actual) else "N/A",
                        delta=f"{auc_actual - auc_baseline:.3f}" if pd.notna(auc_actual) and pd.notna(auc_baseline) else None
                    )
                
                with col4:
                    st.metric(
                        "AUC Baseline",
                        f"{auc_baseline:.3f}" if pd.notna(auc_baseline) else "N/A"
                    )
                
                # Gráfico de evolución
                if pd.notna(f1_actual) and pd.notna(auc_actual) and pd.notna(f1_baseline) and pd.notna(auc_baseline):
                    fig_data = {
                        'Métrica': ['Precisión Ocurrencia (F1)', 'Precisión Volumen (AUC)'],
                        'Baseline': [f1_baseline, auc_baseline],
                        'Actual': [f1_actual, auc_actual]
                    }
                    df_fig = pd.DataFrame(fig_data).melt('Métrica', var_name='Tipo', value_name='Score')
                    
                    fig = px.bar(
                        df_fig,
                        x='Métrica',
                        y='Score',
                        color='Tipo',
                        barmode='group',
                        text_auto='.3f',
                        title="Comparación Baseline vs. Actual"
                    )
                    fig.update_layout(
                        height=350, 
                        yaxis=dict(range=[0,1]),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#FFFFFF",
                        legend_title_font_color="#FFFFFF",
                        legend_font_color="#FFFFFF"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar alertas si existen
        if os.path.exists(alertas_path):
            st.markdown("---")
            # MODIFICADO: Emoji eliminado
            st.subheader("Historial de Alertas")
            df_alertas = pd.read_csv(alertas_path)
            st.dataframe(df_alertas.style.set_properties(**{'color': '#FFFFFF'}), use_container_width=True, height=300) # Texto blanco
    
    except FileNotFoundError:
        # MODIFICADO: Emoji eliminado
        st.warning(f"""
        **No se encontró el reporte de monitoreo en `{reporte_path}`.**
        
        Para generar el reporte de salud del modelo, ejecuta la **Celda 11** del notebook:
        - Sistema de Re-entrenamiento y Alertas (MLOps)
        
        Esto creará el archivo `reporte_monitoreo.csv` necesario para esta pestaña.
        """)
        
        
        st.info("""
        **¿Qué hace la Celda 11?**
        1. Evalúa el rendimiento actual del modelo vs. baseline
        2. Detecta drift (degradación del modelo)
        3. Re-entrena automáticamente si es necesario
        4. Genera alertas y reportes de monitoreo
        """)