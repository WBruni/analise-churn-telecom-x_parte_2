# --- IMPORTS ---
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import display, HTML
from scipy.stats import chi2_contingency
import ipywidgets as widgets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score


# --- FUNÇÕES ---
# 1. VARIÁVEL ALVO 
def plot_distribuicao_target(df, coluna_target='Churn'):
    """
    Cria um histograma da variável alvo com percentuais e anotação estratégica.
    """
    # 1. Cálculo das porcentagens
    churn_percentages = (df[coluna_target].value_counts(normalize=True) * 100)
    
    # 2. Criação da figura base
    fig = px.histogram(df, x=coluna_target, color=coluna_target)

    # 3. Atualização das traces com percentuais
    for trace in fig.data:
        category = trace.x[0]
        percentage = churn_percentages.get(category, 0.0)
        trace.text = [f"{percentage:.1f}%"]
        trace.textposition = 'outside'
        trace.textfont = dict(color='black')

    # 4. Lógica para encontrar o label de Churn (Sim/Yes/1)
    # Tenta encontrar 'Yes', 'Sim' ou o valor 1 como indicadores de Churn
    posssiveis_labels = ['Yes', 'Sim', 1, '1']
    label_churn = next((l for l in posssiveis_labels if l in churn_percentages.index), churn_percentages.index[0])
    
    contagem_churn = df[coluna_target].value_counts()[label_churn]

    # 5. Adicionando a anotação estratégica
    fig.add_annotation(
        x=label_churn,
        y=contagem_churn,
        text="<b>Público Alvo da Retenção</b>",
        showarrow=False,
        yshift=45,
        xanchor='center',
        font=dict(size=14, color="#EF553B"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#EF553B",
        borderwidth=1,
        borderpad=5
    )

    # 6. Configurações de Layout
    fig.update_layout(
        hoverlabel=dict(font_color="white"),
        width=800,
        height=500,
        title='Distribuição de Churn (Variável Alvo)',
        title_font_size=20,
        title_x=0.5,
        xaxis_title='Status Churn',
        yaxis_title='Quantidade de Clientes',
        xaxis=dict(title_font=dict(size=16)),
        yaxis=dict(title_font=dict(size=16))
    )

    return fig # Retorna o objeto para ser exibido no notebook

# ----- # -----------

# 2. VARIÁVEIS CATEGÓRICAS (Função Geral)
def plot_bloco_eda(df, colunas, titulo_bloco):
    """
    Gera um grid de subplots comparando Volume Absoluto e Taxa de Churn.
    """
    total_registros = len(df)
    n_rows = len(colunas)
    
    # Cálculo dinâmico do y_max para manter a escala dos gráficos de volume igual
    y_max_volume = df[colunas].apply(lambda x: x.value_counts()).max().max() * 1.25

    # Criar títulos dinamicamente para cada rastro
    titulos = []
    for col in colunas:
        titulos.extend([f"Volume: {col}", f"Taxa %: {col}"])

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=titulos,
        horizontal_spacing=0.1, 
        vertical_spacing=max(0.05, 0.2 / n_rows) # Ajusta o espaço vertical conforme o número de linhas
    )

    cores_map = {'No': '#636EFA', 'Yes': '#EF553B'}

    for i, col in enumerate(colunas):
        row = i + 1
        counts = df.groupby([col, 'Churn']).size().unstack(fill_value=0)
        pct_total = (counts / total_registros * 100)
        pct_cat = counts.div(counts.sum(axis=1), axis=0) * 100
        
        for status in ['No', 'Yes']:
            # --- COLUNA 1: VOLUME ---
            fig.add_trace(
                go.Bar(
                    name=f"{status}", x=counts.index, y=counts[status], 
                    marker_color=cores_map[status], 
                    showlegend=(i == 0), # Legenda apenas no primeiro gráfico
                    offsetgroup=status,
                    text=pct_total[status].apply(lambda x: f"{x:.1f}%"),
                    textposition='outside', cliponaxis=False
                ),
                row=row, col=1
            )

            # --- COLUNA 2: TAXA ---
            ancora_texto = 'middle' if status == 'Yes' else 'end' 

            fig.add_trace(
                go.Bar(
                    name=f"Churn {status}", x=pct_cat.index, y=pct_cat[status], 
                    marker_color=cores_map[status], showlegend=False,
                    offsetgroup="stack",
                    text=pct_cat[status].apply(lambda x: f"{x:.1f}%"),
                    textposition='inside',
                    insidetextanchor=ancora_texto,
                    insidetextfont=dict(size=11, color='white'),
                    hovertemplate=f"Taxa: %{{y:.1f}}%<extra>{status}</extra>"
                ),
                row=row, col=2
            )

    # AJUSTES DE LAYOUT
    fig.update_layout(
        hoverlabel=dict(font_color="white"),
        height=350 * n_rows, # Altura dinâmica baseada no número de variáveis
        width=900, 
        title_text=titulo_bloco,
        title_x=0.5, barmode='group',
        title_font_size=22,
        plot_bgcolor='#E5ECF6', paper_bgcolor='white',
        uniformtext=dict(mode='show', minsize=9),
        margin=dict(l=30, r=30, t=100, b=40),
        legend_title_text="Churn"
    )

    # Padronização dos eixos
    for r in range(1, n_rows + 1):
        fig.update_yaxes(range=[0, y_max_volume], row=r, col=1, gridcolor='white', title_text="Qtd", title_font_size=16)
        fig.update_yaxes(range=[0, 105], row=r, col=2, gridcolor='white', title_text="(%)", title_font_size=16)
        fig.update_xaxes(gridcolor='white', row=r, col=1)
        fig.update_xaxes(gridcolor='white', row=r, col=2)

    return fig

# ----- # -----------

# 2.1 VARIÁVEIS CATEGÓRICAS (Perfil Demográfico)
def plot_perfil_demografico(df):
    """
    Gera o bloco EDA demográfico com as anotações estratégicas de Senior e Dependents.
    """
    cols_demo = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents']
    
    # Chama a sua função base que cria os subplots
    # (Supondo que plot_bloco_eda já está no seu .py)
    fig = plot_bloco_eda(df, cols_demo, "EDA: Perfil Demográfico")

    # ANOTAÇÃO 1: Foco em SeniorCitizen
    fig.add_annotation(
        dict(
            x="Yes", y=41.6, 
            xref="x4", yref="y4", # O Plotly identifica o 2º gráfico da 2ª linha como x4/y4
            text="<b>Segmento Crítico Churn</b>",
            showarrow=False, yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#EF553B", borderwidth=1
        )
    )

    # ANOTAÇÃO 2: Foco em Dependents
    fig.add_annotation(
        dict(
            x="No", y=31.2, 
            xref="x8", yref="y8", # Referência ao subplot de Dependents
            text="<b>Risco Dobrado Churn</b>",
            showarrow=False, yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#EF553B", borderwidth=1
        )
    )

    # Ajuste de layout para o título não cortar
    fig.update_layout(margin=dict(t=100))

    return fig

# ----- # -----------

# 2.2 VARIÁVEIS CATEGÓRICAS (Serviços Infraestrutura)
def plot_servicos_infra(df):
    """
    Gera o bloco EDA Serviços Infra com as anotações estratégicas na Fibra Óptica.
    """
    cols_servicos_infra = ['PhoneService', 'MultipleLines', 'InternetService']
    
    # Chama a sua função base que cria os subplots
    # (Supondo que plot_bloco_eda já está no seu .py)
    fig = plot_bloco_eda(df, cols_servicos_infra, "EDA: Serviços Infra")


    # ANOTAÇÃO: Foco em Fiber Optic (Gráfico 6)
    fig.add_annotation(
        dict(
            x="Fiber optic", 
            y=41.8, 
            xref="x6", yref="y6", 
            text="<b>Ponto Crítico Churn</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1,
            borderpad=4
        )
    )
    
    fig.update_layout(margin=dict(t=100))
    
    return fig

# ----- # -----------

# 2.3 VARIÁVEIS CATEGÓRICAS (Serviços Segurança)
def plot_servicos_seg(df):
    """
    Gera o bloco EDA Serviços Segurança com as anotações estratégicas para Segurança Online e Suporte Técnico.
    """
    cols_servicos_seg = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    
    # Chama a sua função base que cria os subplots
    # (Supondo que plot_bloco_eda já está no seu .py)
    fig = plot_bloco_eda(df, cols_servicos_seg, "EDA: Serviços Segurança")


    # ANOTAÇÃO 1: OnlineSecurity (Gráfico 2 - Taxa %)
    fig.add_annotation(
        dict(
            x="No", 
            y=31.2, 
            xref="x2", yref="y2", 
            text="<b>Ausência de Segurança Online</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1
        )
    )
    
    # ANOTAÇÃO 2: TechSupport (Gráfico 8 - Taxa %)
    fig.add_annotation(
        dict(
            x="No", 
            y=31.1, 
            xref="x8", yref="y8", 
            text="<b>Falta de Suporte Técnico</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1
        )
    )
    
    fig.update_layout(margin=dict(t=100))
    
    return fig

# ----- # -----------

# 2.4 VARIÁVEIS CATEGÓRICAS (Serviços Entretenimento)
def plot_servicos_entret(df):
    """
    Gera o bloco EDA Serviços Entretenimento com as anotações estratégicas para StreamingTV e Streaming Filmes.
    """
    cols_servicos_entret = ['StreamingTV', 'StreamingMovies']
    
    # Chama a sua função base que cria os subplots
    # (Supondo que plot_bloco_eda já está no seu .py)
    fig = plot_bloco_eda(df, cols_servicos_entret, "EDA: Entretenimento")

    
    # ANOTAÇÃO: Foco em StreamingTV (Gráfico 1 - Taxa %)
    fig.add_annotation(
        dict(
            x="Yes", 
            y=30.1, 
            xref="x2", yref="y2", 
            text="<b>Streaming TV não retém clientes</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1
        )
    )
    
    
    # ANOTAÇÃO: Foco em StreamingFilmes (Gráfico 2 - Taxa %)
    fig.add_annotation(
        dict(
            x="Yes", 
            y=30.1, 
            xref="x4", yref="y4", 
            text="<b>Streaming Fimes não retém clientes</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1
        )
    )
    
    
    
    fig.update_layout(margin=dict(t=80))
    
    return fig
        
# ----- # -----------    
    
# 2.5 VARIÁVEIS CATEGÓRICAS (Contratual/ Financeiro)
def plot_perfil_contratual_financeiro(df):
    """
    Gera o bloco EDA Contratual e Financeiro com as anotações estratégicas para Controto e Método de Pagamento.
    """
    cols_contratual = ['Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Chama a sua função base que cria os subplots
    # (Supondo que plot_bloco_eda já está no seu .py)
    fig = plot_bloco_eda(df, cols_contratual, "EDA: Contratos e Faturamento")
    

    # ANOTAÇÃO 1: Contract (Gráfico 2 - Taxa %)
    fig.add_annotation(
        dict(
            x="Month-to-month", 
            y=25, 
            xref="x2", yref="y2", 
            text="<b>Baixa Fidelidade</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1
        )
    )
    
    # ANOTAÇÃO 2: PaymentMethod (Gráfico 6 - Taxa %)
    fig.add_annotation(
        dict(
            x="Electronic check", 
            y=25, 
            xref="x6", yref="y6", 
            text="<b>Método com Maior Evasão</b>",
            showarrow=False,
            yshift=15,
            font=dict(color="#EF553B", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EF553B",
            borderwidth=1
        )
    )
    
    fig.update_layout(margin=dict(t=100), height=900)
    
    return fig

# ----- # -----------  

# 3. VARIÁVEIS NUMÉRICAS
def plot_boxplots_numericos(df, coluna_target='Churn'):
    """
    Cria uma grade de boxplots para todas as variáveis numéricas do dataframe,
    comparando por Churn e adicionando anotações automáticas das medianas.
    """
    # 1. Filtra colunas numéricas
    colunas_num = df.select_dtypes(include=['number']).columns.tolist()

    # 2. Configura a grade
    n_cols = 2
    n_rows = (len(colunas_num) + 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        subplot_titles=colunas_num,
        horizontal_spacing=0.1, 
        vertical_spacing=0.15
    )

    # 3. Loop automático para Gráficos e Anotações
    for i, col in enumerate(colunas_num):
        row = (i // n_cols) + 1
        c_idx = (i % n_cols) + 1
        
        # Identificadores de eixo corretos
        xref_idx = f"x{i+1}" if i > 0 else "x"
        yref_idx = f"y{i+1}" if i > 0 else "y"

        # Cálculo das medianas (ajustado para aceitar labels Sim/Yes ou No/Não)
        label_no = 'No' if 'No' in df[coluna_target].unique() else 'Não'
        label_yes = 'Yes' if 'Yes' in df[coluna_target].unique() else 'Sim'

        m_no = df[df[coluna_target] == label_no][col].median()
        m_yes = df[df[coluna_target] == label_yes][col].median()

        # Formatação de etiquetas
        prefix = "R$ " if "Charges" in col else ""
        suffix = " meses" if col == "Tenure" else ""

        # Adicionando Traces (Boxplots)
        fig.add_trace(
            go.Box(y=df[df[coluna_target] == label_no][col], name=label_no, marker_color='#636EFA', showlegend=(i==0)),
            row=row, col=c_idx
        )
        fig.add_trace(
            go.Box(y=df[df[coluna_target] == label_yes][col], name=label_yes, marker_color='#EF553B', showlegend=(i==0)),
            row=row, col=c_idx
        )

        # Anotação para o grupo "No" (Lado Esquerdo)
        fig.add_annotation(
            dict(
                x=label_no, y=m_no, xref=xref_idx, yref=yref_idx,
                text=f"<b>{prefix}{m_no:.2f}{suffix}</b>",
                showarrow=True, arrowhead=2, arrowcolor="#636EFA",
                ax=-50, ay=0, font=dict(color="#636EFA", size=10),
                bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#636EFA", borderwidth=1
            )
        )

        # Anotação para o grupo "Yes" (Lado Direito)
        fig.add_annotation(
            dict(
                x=label_yes, y=m_yes, xref=xref_idx, yref=yref_idx,
                text=f"<b>{prefix}{m_yes:.2f}{suffix}</b>",
                showarrow=True, arrowhead=2, arrowcolor="#EF553B",
                ax=50, ay=0, font=dict(color="#EF553B", size=10),
                bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#EF553B", borderwidth=1
            )
        )

    # 4. Configurações de Layout
    fig.update_layout(
        hoverlabel=dict(font_color="white"),
        height=700, 
        width=1000,
        title_text="EDA: Variáveis Numéricas", 
        title_x=0.5,
        showlegend=True,
        legend_title_text="Churn",
        margin=dict(l=50, r=50, t=100, b=40)
    )

    # Ajuste dos eixos Y
    for i, col in enumerate(colunas_num):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1
        title_y = "($)" if "Charges" in col else "Qtd"
        fig.update_yaxes(title_text=title_y, row=r, col=c)

    return fig

# ----- # -----------

# 4. VERIFICAÇÃO OUTLIERS
def exibir_analise_outliers(df, apenas_churn=False, coluna_target='Churn'):
    """
    Função unificada para análise de outliers.
    Argumentos:
    - apenas_churn: Se True, filtra apenas clientes com Churn. Se False, analisa o dataset todo.
    - coluna_target: Nome da coluna de Churn.
    """
    
    # 1. Define o título e aplica o filtro se necessário
    titulo = "<h3>📊 Análise de Outliers (Apenas Clientes com Churn)</h3>" if apenas_churn else "<h3>🔍 Resumo Geral de Outliers (Dataset Completo)</h3>"
    
    df_processar = df[df[coluna_target].isin(['Yes', 'Sim'])] if apenas_churn else df

    # 2. Selecionamos apenas as colunas numéricas
    df_num = df_processar.select_dtypes(include='number')

    # 3. Lista para armazenar os resultados
    lista_contagem = []

    for col in df_num.columns:
        Q1 = df_num[col].quantile(0.25)
        Q3 = df_num[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        
        outliers = df_num[(df_num[col] < lim_inf) | (df_num[col] > lim_sup)]
        
        lista_contagem.append({
            "Variável": col,
            "Qtd de Outliers": len(outliers),
            "Percentual (%)": f"{(len(outliers) / len(df_processar) * 100):.2f}%",
            "Limite Inf": round(lim_inf, 2),
            "Limite Sup": round(lim_sup, 2)
        })

    # 4. Criação e exibição da tabela
    df_resumo = pd.DataFrame(lista_contagem)
    html_output = titulo + df_resumo.to_html(index=False, classes='table table-striped table-hover', justify='left')
    
    display(HTML(html_output))
    #return df_resumo
        
# ----- # -----------

# 5. TENDÊNCIA NUMÉRICA
def plot_num_eda_kde(df, colunas_num, titulo_bloco):
    n_rows = len(colunas_num)
    # Criamos o subplot
    fig = make_subplots(rows=n_rows, cols=1, vertical_spacing=0.08,
                        subplot_titles=[f"Densidade de Churn: {c}" for c in colunas_num])

    cores = ['#636EFA', '#EF553B'] # Azul e Vermelho

    for i, col in enumerate(colunas_num):
        # Filtramos os dados e removemos nulos para não quebrar o gráfico
        df_clean = df[[col, 'Churn']].dropna()
        
        hist_data = [df_clean[df_clean['Churn'] == 'No'][col], 
                     df_clean[df_clean['Churn'] == 'Yes'][col]]
        group_labels = ['No', 'Yes']

        # Criamos o gráfico de densidade (KDE)
        # show_hist=False remove as barras e deixa só a curva suave
        fig_kde = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=cores)

        # Adicionamos as curvas ao subplot principal
        for trace in fig_kde.data:
            fig.add_trace(trace, row=i+1, col=1)
            # Mostra legenda apenas no primeiro subplot
            if i > 0:
                fig.data[-1].showlegend = False

    fig.update_layout(
        hoverlabel=dict(font_color="white"),
        height=400 * n_rows, width=900,
        title_text=titulo_bloco,
        title_x=0.5, barmode='group',
        title_font_size=22,
        plot_bgcolor='#E5ECF6', paper_bgcolor='white',
        legend_title_text="Churn"
    )
    
    for r in range(1, n_rows + 1):
        fig.update_xaxes(gridcolor='white', row=1, col=1, title_text="Months", title_font_size=16)
        fig.update_xaxes(gridcolor='white', row=2, col=1, title_text="($)", title_font_size=16)
        fig.update_xaxes(gridcolor='white', row=3, col=1, title_text="($)", title_font_size=16)

        
        # 1. Anotação para Tenure (Eixo 1)
    fig.add_annotation(
        x=3, y=0.04, xref="x1", yref="y1",
        text="<b>Churn Precoce</b>",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        font=dict(color="#EF553B"), bgcolor="white", bordercolor="#EF553B"
    )
    
    # 2. Anotação para ChargesMonthly (Eixo 2)
    fig.add_annotation(
        x=72, y=0.015, xref="x2", yref="y2",
        text="<b>Alta Sensibilidade a Preço</b>",
        showarrow=True, arrowhead=2, ax=-120, ay=-40,
        font=dict(color="#EF553B"), bgcolor="white", bordercolor="#EF553B"
    )
    
    # 3. Anotação para ChargesTotal (Eixo 3)
    fig.add_annotation(
        x=320, y=0.00045, xref="x3", yref="y3",
        text="<b>Perda de Clientes Novos</b>",
        showarrow=True, arrowhead=2, ax=50, ay=-30,
        font=dict(color="#EF553B"), bgcolor="white", bordercolor="#EF553B"
    )
    
    return fig

# ----- # -----------

# 6. MATRIZ CORRELAÇÂO NUMÉRICA
def plot_heatmap_correlacao(df, coluna_target='Churn'):
    """
    Gera um heatmap de correlação para as variáveis numéricas,
    incluindo a variável alvo mapeada e anotações de destaque.
    """
    # 1. Preparação dos dados: Cópia para não alterar o df original do notebook
    df_corr = df.copy()
    
    # Mapeamento dinâmico para garantir que o Churn seja numérico no cálculo
    if df_corr[coluna_target].dtype == 'object':
        mapping = {'No': 0, 'Yes': 1, 'Não': 0, 'Sim': 1}
        df_corr[coluna_target] = df_corr[coluna_target].map(mapping)

    # 2. Calculando a correlação
    corr_matrix = df_corr.corr(numeric_only=True)

    # 3. Escala de cores personalizada (Sua paleta azul-branco-vermelho)
    custom_rb_weaker_red = [
        [0.0, 'rgb(99, 102, 255)'],    # Azul escuro
        [0.5, 'rgb(232, 232, 232)'],  # Branco
        [1.0, 'rgb(239, 85, 59)']     # Vermelho mais fraco
    ]

    # 4. Criação do Heatmap
    fig = px.imshow(
        corr_matrix, 
        text_auto='.2f', 
        color_continuous_scale=custom_rb_weaker_red,
        title="EDA: Matriz de Correlação (Variáveis Numéricas)"
    )

    # 5. Configurações de Layout
    fig.update_layout(
        width=800, height=500,
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=150, t=80, b=50) # Aumentei a margem direita para a anotação
    )

    # 6. Retângulo de destaque (Ajustado para as coordenadas da matriz)
    fig.add_shape(
        type="rect",
        x0=1.5, y0=2.5, x1=2.5, y1=3.5,
        line=dict(color="yellow", width=3),
        xref='x', yref='y'
    )

    # 7. Anotação explicativa lateral
    fig.add_annotation(
        text="<b>Atenção:</b> ChargesMonthly e ChargesDaily<br>possuem alta correlação (redundância).",
        xref="paper", yref="paper",
        x=1.23, y=0.29, # Ajustado para a lateral
        showarrow=False,
        font=dict(color="#2C3E50", size=11),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.9)", 
        bordercolor="#2C3E50", 
        borderwidth=1
    )

    fig.update_traces(textfont_size=14)

    return fig

# ----- # -----------

# 7. TABELA DECISÂO
def rank_categoricas_completo(df_original, target='Churn'):
    df_temp = df_original.copy()
    
    # Converter binários numéricos para objeto para o teste
    for col in df_temp.select_dtypes(include=['number']).columns:
        if df_temp[col].nunique() <= 2:
            df_temp[col] = df_temp[col].astype(object)

    colunas_cat = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
    if target in colunas_cat: colunas_cat.remove(target)
    
    resultados = []
    
    for col in colunas_cat:
        df_loop = df_temp[[col, target]].dropna()
        contingency_table = pd.crosstab(df_loop[col], df_loop[target])
        
        # chi2 é o Chi-Score (estatística do teste)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        resultados.append({
            'Variável': col,
            'Chi-Score': round(chi2, 2), # Adicionado aqui
            'P-Value': p,
            'Importância': 'Alta' if p < 0.01 else ('Média' if p < 0.05 else 'Baixa/Nula'),
            'Manter na Pipeline?': 'Sim' if p < 0.05 else 'Não'
        })
    
    # Ordenamos pelo Chi-Score de forma decrescente (o mais forte no topo)
    return pd.DataFrame(resultados).sort_values(by='Chi-Score', ascending=False)

# ----- # -----------

# 8. AVALIAÇÂO PERFORMANCE MODELOS (Regressão Logística, Random Forest)
def display_side_by_side(y_test, y_pred_norm, y_pred_bal, modelo_nome=""):
    # 1. Gerar e formatar os DataFrames
    def preparar_df(y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().round(2)
        # Renomear para ficar amigável
        mapping = {'0': 'No', '1': 'Yes', 'accuracy': 'ACURÁCIA GERAL'}
        df.rename(index=mapping, inplace=True)
        return df.loc[['No', 'Yes', 'ACURÁCIA GERAL']]

    df_n = preparar_df(y_pred_norm)
    df_b = preparar_df(y_pred_bal)

    # 2. Criar o HTML para os títulos e tabelas
    # O estilo CSS 'margin' ajuda a separar as tabelas
    estilo_tabela = "style='display: inline-block; margin-right: 50px;'"
    
    html_norm = f"<div {estilo_tabela}><h4>{modelo_nome} (Normal)</h4>{df_n.to_html(classes='table table-striped')}</div>"
    html_bal = f"<div {estilo_tabela}><h4>{modelo_nome} (Balanceado)</h4>{df_b.to_html(classes='table table-striped')}</div>"

    # 3. Exibir lado a lado
    display(HTML(f"<div style='display: flex;'>{html_norm}{html_bal}</div>"))

# ----- # -----------

# 9. MATRIZ DE CONFUSÂO (Regressão Logística, Random Forest, SVM)
def plot_comparacao_matrizes(y_real, y_pred_normal, y_pred_bal, titulo_modelo=""):
    """
    Gera duas matrizes de confusão lado a lado para comparar versões Normal vs Balanceada.
    """
    # 1. Gerar as matrizes numéricas
    conf_simples = confusion_matrix(y_real, y_pred_normal)
    conf_bal = confusion_matrix(y_real, y_pred_bal)

    # 2. Configurações Visuais Padrão
    x_labels = ['No', 'Yes']
    y_labels = ['No', 'Yes']
    custom_colors = [[0.0, 'rgb(99, 102, 255)'], [0.5, 'rgb(232, 232, 232)'], [1.0, 'rgb(239, 85, 59)']]
    color_matrix = [[0, 0.5], [0.5, 1]]

    # 3. Criar Subplots
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f'{titulo_modelo} (Normal)', f'{titulo_modelo} (Balanceado)'),
        horizontal_spacing=0.15
    )

    # 4. Adicionar Matriz Normal
    fig.add_trace(
        go.Heatmap(
            z=color_matrix, x=x_labels, y=y_labels[::-1],
            colorscale=custom_colors, showscale=False,
            text=conf_simples[::-1], texttemplate="%{text}", 
            textfont={"size": 16, "color": "black"}
        ), row=1, col=1
    )

    # 5. Adicionar Matriz Balanceada
    fig.add_trace(
        go.Heatmap(
            z=color_matrix, x=x_labels, y=y_labels[::-1],
            colorscale=custom_colors, showscale=False,
            text=conf_bal[::-1], texttemplate="%{text}", 
            textfont={"size": 16, "color": "black"}
        ), row=1, col=2
    )

    # 6. Ajustes de Layout (Fontes 16 e X embaixo)
    fig.update_layout(
        width=1000, height=500,
        title_text=f"Comparação de Performance: {titulo_modelo}",
        title_x=0.5, title_font_size=20,
        margin=dict(l=100, r=50, t=100, b=100)
    )

    fig.update_xaxes(title_text="Previsto", title_font_size=16, tickfont_size=16, side='bottom')
    fig.update_yaxes(title_text="Real", title_font_size=16, tickfont_size=16)

    return fig

# ----- # -----------

# 10. GRÁFICOS COEFICIENTES (Regressão Logística, )
def plot_coeficientes_regressao(modelo, pre_processamento):
    """
    Extrai os coeficientes da Regressão Logística, mapeia com os nomes das colunas
    pós-processamento e gera um gráfico de barras horizontal com destaques.
    """
    # 1. Pegar os coeficientes do modelo
    # A Regressão Logística guarda os pesos em .coef_
    coefs = modelo.coef_[0] 
    colunas_nomes = pre_processamento.get_feature_names_out()

    # 2. Criar DataFrame e ordenar
    df_coef = pd.DataFrame({'Variável': colunas_nomes, 'Coeficiente': coefs})
    df_coef = df_coef.sort_values(by='Coeficiente', ascending=True)

    # Paleta de cores personalizada (Azul-Branco-Vermelho)
    custom_colors = [
        [0.0, 'rgb(99, 102, 255)'], 
        [0.5, 'rgb(232, 232, 232)'], 
        [1.0, 'rgb(239, 85, 59)']
    ]

    # 3. Criação do gráfico
    fig = px.bar(
        df_coef, x='Coeficiente', y='Variável',
        orientation='h',
        title="Coeficientes da Regressão Logística (Impacto Direto)",
        color='Coeficiente',
        color_continuous_scale=custom_colors,
        text_auto='.3f'
    )

    # 4. Destaque para a Fibra Óptica (O Vilão)
    # Verificamos se a coluna existe para evitar erro caso o pre_processamento mude
    if "InternetService_Fiber optic" in colunas_nomes:
        val_fiber = df_coef.loc[df_coef['Variável'] == "InternetService_Fiber optic", 'Coeficiente'].values[0]
        fig.add_annotation(
            x=val_fiber, y="InternetService_Fiber optic",
            text="<b>Principal Preditor de Churn</b>",
            showarrow=True, arrowhead=2, arrowcolor="red",
            ax=85, ay=0, font=dict(color="red", size=11),
            bgcolor="white", bordercolor="red"
        )

    # 5. Destaque para o Contrato de 2 Anos (A Solução)
    if "Contract_Two year" in colunas_nomes:
        val_contract = df_coef.loc[df_coef['Variável'] == "Contract_Two year", 'Coeficiente'].values[0]
        fig.add_annotation(
            x=val_contract, y="Contract_Two year",
            text="<b>Maior Força de Retenção</b>",
            showarrow=True, arrowhead=2, arrowcolor="blue",
            ax=-80, ay=0, font=dict(color="blue", size=11),
            bgcolor="white", bordercolor="blue"
        )

    # 6. Configurações de Layout
    fig.update_layout(
        hoverlabel=dict(font_color="white"),
        width=900, height=700, 
        title_x=0.5, 
        margin=dict(l=200, r=150, t=100, b=50), 
        title_font_size=22
    )

    fig.update_xaxes(title_font_size=16)
    fig.update_yaxes(title_font_size=16)

    return fig

# ----- # -----------

# 11. FEAUTURE IMPORTANCE (Função Geral)
def plot_feature_importance(modelo, colunas, titulo="Top 10 Variáveis Mais Influentes"):
    """
    Gera um gráfico de barras horizontal apresentando a importância das variáveis 
    (Feature Importance) baseada no ganho de informação do modelo.
    """
    # 1. Extrair as importâncias e criar um DataFrame
    importancias = modelo.feature_importances_
    df_feat = pd.DataFrame({
        'Variável': colunas,
        'Importância': importancias
    }).sort_values(by='Importância', ascending=True).tail(10) # Foca no Top 10

    # Paleta de cores consistente com a identidade do projeto Telecom X
    custom_colors = [
        [0.0, 'rgb(99, 102, 255)'], 
        [0.5, 'rgb(232, 232, 232)'], 
        [1.0, 'rgb(239, 85, 59)']
    ]
    
    # 2. Criar o gráfico de barras horizontal usando Plotly Express
    fig = px.bar(
        df_feat, 
        x='Importância', 
        y='Variável', 
        orientation='h',
        title=titulo,
        text_auto='.3f',
        color='Importância',
        color_continuous_scale=custom_colors
    )

    # 3. Ajustes de Layout e Tipografia
    fig.update_layout(
        width=800, 
        height=500,
        xaxis=dict(title='Ganho de Informação', tickfont_size=14, title_font_size=16),
        yaxis=dict(title='', tickfont_size=14),
        title_x=0.5,
        title_font_size=20,
        margin=dict(l=150, r=50, t=80, b=50) # Garante que os nomes das variáveis não sejam cortados
    )
    
    return fig

# ----- # -----------

# 11.1 FEAUTURE IMPORTANCE (Random Forest)
def plot_feat_importance_rf(modelo, nomes_colunas):
    """
    Gera o gráfico de Feature Importance para o Random Forest 
    com anotações estratégicas sobre as variáveis contínuas.
    """
    
    # CORREÇÃO: Usamos os argumentos passados (modelo e nomes_colunas)
    # em vez de nomes fixos como 'rf_simples'
    fig = plot_feature_importance(modelo, nomes_colunas, "Importância das Variáveis - Random Forest")
    
    # CORREÇÃO: Para o 'x' não ficar fixo, buscamos o valor real no modelo
    # Isso garante que a seta sempre aponte para o final da barra
    importancias = modelo.feature_importances_
    df_temp = pd.DataFrame({'Var': nomes_colunas, 'Imp': importancias})
    
    try:
        val_charges = df_temp.loc[df_temp['Var'] == "ChargesMonthly", 'Imp'].values[0]
        
        fig.add_annotation(
            dict(
                x=val_charges, # Agora o x é dinâmico!
                y="ChargesMonthly",
                text="<b>Dominância Numérica:</b><br>As variáveis contínuas são<br>os principais critérios de divisão.",
                showarrow=True,
                arrowhead=2,
                ax=80, ay=80,
                font=dict(color="#2C3E50", size=11),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#2C3E50",
                borderwidth=1
            )
        )
    except IndexError:
        print("Aviso: Variável 'ChargesMonthly' não encontrada para anotação.")

    fig.update_layout(margin=dict(r=150), 
                      hoverlabel=dict(font_color="white"))
    
    return fig

# ----- # -----------

# 11.2 FEAUTURE IMPORTANCE (XGBoost)
def plot_feat_importance_xg(modelo, nomes_colunas):
    """
    Gera o gráfico de Feature Importance para o XGBoost 
    com anotações estratégicas sobre as variáveis contínuas.
    """
    
    # CORREÇÃO: Usamos os argumentos passados (modelo e nomes_colunas)
    # em vez de nomes fixos como 'rf_simples'
    fig = plot_feature_importance(modelo, nomes_colunas, "Importância das Variáveis - XGBoost")
    
    # CORREÇÃO: Para o 'x' não ficar fixo, buscamos o valor real no modelo
    # Isso garante que a seta sempre aponte para o final da barra
    importancias = modelo.feature_importances_
    df_temp = pd.DataFrame({'Var': nomes_colunas, 'Imp': importancias})
    
    try:
        val_charges = df_temp.loc[df_temp['Var'] == "Contract_Two year", 'Imp'].values[0]

        fig.add_annotation(
            dict(
                x=val_charges,
                y="Contract_Two year",
                text="<b>Fator Decisivo:</b> O tipo de contrato é o<br>maior preditor de retenção para o XGBoost.<br><b>Contrato Bienal:</b> 0.362",
                showarrow=True,
                arrowhead=2,
                ax=80, ay=80, # Posicionamento da caixa de texto
                font=dict(color="#2C3E50", size=11),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#2C3E50",
                borderwidth=1
            )
        )
    except IndexError:
        print("Aviso: Variável 'Contract_Two year' não encontrada para anotação.")
    
    fig.update_layout(margin=dict(r=150), 
                      hoverlabel=dict(font_color="white")) 
    
    return fig

# ----- # -----------

# 12 AVALIAÇÂO PERFORMANCE (XGBoost)
def display_side_by_side_xgb(y_test_num, y_pred_xgb, modelo_nome="XGBoost"):
    """
    Refatorada para o XGBoost (que usa 0 e 1).
    Exibe a tabela de métricas formatada.
    """
    def preparar_df(y_pred):
        # Gera o relatório (precisa ser numérico para bater com o y_test_num)
        report = classification_report(y_test_num, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().round(2)
        
        # Mapeamento: O classification_report usa strings '0' e '1' para as chaves
        # Renomeamos para o padrão do projeto (No/Yes)
        mapping = {'0': 'No', '1': 'Yes', 'accuracy': 'ACURÁCIA GERAL'}
        df.rename(index=mapping, inplace=True)
        
        # Retorna apenas as linhas principais para a tabela não ficar gigante
        return df.loc[['No', 'Yes', 'ACURÁCIA GERAL']]

    # Como o XGBoost geralmente é treinado já com foco em balanceamento (scale_pos_weight),
    # aqui geramos a tabela dele.
    df_xgb = preparar_df(y_pred_xgb)

    # Criar o HTML para a tabela
    estilo_tabela = "style='display: inline-block; margin-right: 50px; font-family: sans-serif;'"
    
    html_xgb = f"""
    <div {estilo_tabela}>
        <h4>{modelo_nome} Final</h4>
        {df_xgb.to_html(classes='table table-striped')}
    </div>
    """

    # Exibir
    display(HTML(f"<div style='display: flex;'>{html_xgb}</div>"))

# ----- # -----------

# 13. MATRIZ DE CONFUSÂO (XGBoost)
def plot_matriz_single_xgboost(y_real, y_pred, titulo_modelo="XGBoost"):
    """
    Gera uma matriz de confusão única no padrão Plotly do projeto,
    ajustada para os dados numéricos do XGBoost.
    """
    # 1. Gerar a matriz numérica
    # Note: y_real deve ser o y_test_xgb (0 e 1)
    conf = confusion_matrix(y_real, y_pred)

    # 2. Configurações Visuais (IDÊNTICAS ao seu padrão)
    x_labels = ['No', 'Yes']
    y_labels = ['No', 'Yes']
    # Escala de cores personalizada (Roxo -> Cinza -> Laranja)
    custom_colors = [[0.0, 'rgb(99, 102, 255)'], [0.5, 'rgb(232, 232, 232)'], [1.0, 'rgb(239, 85, 59)']]
    color_matrix = [[0, 0.5], [0.5, 1]]

    # 3. Criar a Figura (Single Plot)
    fig = go.Figure()

    # 4. Adicionar a Matriz
    fig.add_trace(
        go.Heatmap(
            z=color_matrix, 
            x=x_labels, 
            y=y_labels[::-1], # Inverte para o 'No' ficar no topo
            colorscale=custom_colors, 
            showscale=False,
            text=conf[::-1], # Inverte os dados para alinhar com o eixo Y
            texttemplate="%{text}", 
            textfont={"size": 18, "color": "black"} # Fonte levemente maior para destaque
        )
    )

    # 5. Ajustes de Layout (Mantendo fontes e tamanhos do seu padrão)
    fig.update_layout(
        width=600, height=500,
        title_text=f"Performance Final: {titulo_modelo}",
        title_x=0.5, title_font_size=20,
        margin=dict(l=100, r=50, t=100, b=100),
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Previsto", title_font_size=16, tickfont_size=16, side='bottom')
    fig.update_yaxes(title_text="Real", title_font_size=16, tickfont_size=16)

    return fig

# ----- # -----------

# 14. COMPARAÇÂO RESULTADOS MODELOS
def extrair_metricas(y_true, y_pred, nome_modelo):
    """
    Função auxiliar interna para calcular métricas individuais.
    Lida com a conversão de tipos entre (Yes/No) e (0/1).
    """
    # Verifica se há necessidade de mapeamento (caso o predito seja 0/1 e o real seja Yes/No)
    # Pegamos o primeiro elemento para checar o tipo
    primeiro_pred = y_pred[0]
    primeiro_real = y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0]

    if isinstance(primeiro_pred, (int, np.integer)) and isinstance(primeiro_real, str):
        y_true_mod = pd.Series(y_true).map({'No': 0, 'Yes': 1})
        pos_label = 1
    else:
        y_true_mod = y_true
        # Se os dados forem strings, o alvo de churn é 'Yes', se for numérico, é 1
        pos_label = 'Yes' if isinstance(primeiro_real, str) else 1

    return {
        'Modelo': nome_modelo,
        'Acurácia': accuracy_score(y_true_mod, y_pred),
        'Recall (Churn)': recall_score(y_true_mod, y_pred, pos_label=pos_label),
        'Precisão (Churn)': precision_score(y_true_mod, y_pred, pos_label=pos_label),
        'F1-Score': f1_score(y_true_mod, y_pred, pos_label=pos_label)
    }

def exibir_ranking_modelos(lista_modelos_preds, y_test):
    """
    Gera um DataFrame estilizado comparando múltiplos modelos de classificação.
    O ranking é ordenado pelo Recall, focado na estratégia de retenção (Churn).

    Argumentos:
        lista_modelos_preds (list): Lista de tuplas no formato [ (predicao, "Nome do Modelo"), ... ]
        y_test: Os valores reais de teste (Target).

    Retorna:
        pd.DataFrame: O DataFrame de comparação (também exibe a tabela estilizada no Colab).
    """
    comparativo = []
    
    for y_pred, nome in lista_modelos_preds:
        metricas = extrair_metricas(y_test, y_pred, nome)
        comparativo.append(metricas)

    # Criando o DataFrame
    df_final = pd.DataFrame(comparativo).set_index('Modelo').sort_values(by='Recall (Churn)', ascending=False)

    # Exibição Visual (Específico para Notebooks)
    display(HTML("<h3>🏆 Ranking Final de Modelos (Foco em Retenção)</h3>"))
    
    # Aplicando estilização: Verde para o melhor (max), Vermelho para o pior (min)
    estilo = df_final.style.highlight_max(color='#1B9E77', axis=0) \
                           .highlight_min(color='#EF553B', axis=0) \
                           .format("{:.2%}")
    
    display(estilo)
    
    return df_final

# ----- # -----------

# 15. CURVA ROC (Função Geral)
def plot_comparacao_roc(modelos_lista, X_test, y_test):
    """
    Gera uma Curva ROC comparativa para múltiplos modelos.
    
    Argumentos:
        modelos_lista (list): Lista de tuplas [(nome, objeto_modelo), ...]
        X_test: Dados de teste transformados.
        y_test: Target real.
    """
    fig = go.Figure()
    
    # Linha de referência (Chute aleatório)
    fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)

    for nome, modelo in modelos_lista:
        # 1. Obter as probabilidades (classe positiva)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        
        # 2. Ajustar y_test para numérico se necessário
        y_test_num = y_test.map({'No': 0, 'Yes': 1}) if y_test.dtype == 'O' else y_test
        
        # 3. Calcular FPR, TPR e AUC
        fpr, tpr, _ = roc_curve(y_test_num, y_proba)
        auc_score = roc_auc_score(y_test_num, y_proba)
        
        # 4. Adicionar a linha ao gráfico
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{nome} (AUC: {auc_score:.3f})', mode='lines'))

    fig.update_layout(
        hoverlabel=dict(font_color="white"),
        title='Curva ROC Comparativa dos Modelos',
        xaxis_title='Taxa de Falso Positivo (1 - Especificidade)',
        yaxis_title='Taxa de Verdadeiro Positivo (Recall)',
        width=800, height=500,
        title_font_size=20,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        title_x=0.5
    )
    
    return fig

# ----- # -----------

# 15.1 CURVA ROC COMPARATIVA
def plot_roc_comparativa(modelos_lista, X_test, y_test):
    """
    Chama a Curva ROC e adiciona o insight específico sobre 
    o melhor poder preditivo observado no projeto Telecom X.
    """
    fig = plot_comparacao_roc(modelos_lista, X_test, y_test)
    
    # Adicionando o insight estratégico
    fig.add_annotation(
        dict(
            x=0.3, y=0.82,
            xref="x", yref="y",
            text="<b>Melhor Desempenho:</b><br>XGBoost e Regressão Logística<br>possuem o maior poder preditivo.",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-90,
            font=dict(color="#2C3E50", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#2C3E50",
            borderwidth=1
        )
    )
    
    return fig