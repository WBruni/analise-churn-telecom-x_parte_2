import streamlit as st
import pandas as pd
import joblib
import os

# Configuração visual
st.set_page_config(page_title="TelecomX - Simulador de Churn", page_icon="📡")

st.title("📡 Simulador de Churn TelecomX")
st.markdown("Previsão de Churn baseada em modelos de Machine Learning.")

# 1. Carregar os artefatos
@st.cache_resource
def carregar_arquivos():
    modelo = joblib.load('modelo_churn_xgb.pkl')
    processador = joblib.load('pre_processador.pkl')
    return modelo, processador

try:
    model, processor = carregar_arquivos()
except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}. Verifique se 'modelo_xgb.pkl' e 'pre_processador.pkl' estão na mesma pasta.")
    st.stop()

# 2. Interface Lateral para entrada de dados
st.sidebar.header("📋 Perfil do Cliente")

# Inputs baseados nas 16 colunas que o seu processador espera
tenure = st.sidebar.slider("Meses de contrato (Tenure)", 0, 72, 24)
charges = st.sidebar.number_input("Valor da Mensalidade (ChargesMonthly)", 0.0, 200.0, 65.0)
senior = st.sidebar.selectbox("Idoso (SeniorCitizen)?", ["No", "Yes"])
partner = st.sidebar.selectbox("Possui Parceiro?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Possui Dependentes?", ["No", "Yes"])
lines = st.sidebar.selectbox("Múltiplas Linhas?", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Serviço de Internet", ["DSL", "Fiber optic", "No"])
security = st.sidebar.selectbox("Segurança Online", ["No", "Yes", "No internet service"])
backup = st.sidebar.selectbox("Backup Online", ["No", "Yes", "No internet service"])
protection = st.sidebar.selectbox("Proteção de Dispositivo", ["No", "Yes", "No internet service"])
support = st.sidebar.selectbox("Suporte Técnico", ["No", "Yes", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Tipo de Contrato", ["One year", "Month-to-month", "Two year"])
paperless = st.sidebar.selectbox("Fatura Digital?", ["No", "Yes"])
payment = st.sidebar.selectbox("Forma de Pagamento", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 3. Criar o DataFrame com a ordem EXATA das colunas originais do seu X_train
# (Ajuste a ordem abaixo se o seu X_train original tiver uma ordem diferente)
dados_dic = {
    'Tenure': [tenure],
    'ChargesMonthly': [charges],
    'SeniorCitizen': [senior],
    'Partner': [partner],
    'Dependents': [dependents],
    'MultipleLines': [lines],
    'InternetService': [internet],
    'OnlineSecurity': [security],
    'OnlineBackup': [backup],
    'DeviceProtection': [protection],
    'TechSupport': [support],
    'StreamingTV': [tv],
    'StreamingMovies': [movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless],
    'PaymentMethod': [payment]
}

df_usuario = pd.DataFrame(dados_dic)

# 4. Processamento
try:
    # Transforma as 16 colunas originais nas 20 colunas que o modelo espera
    dados_processados = processor.transform(df_usuario)
    
    # Criar um DataFrame com o resultado para o modelo não reclamar de falta de nomes
    nomes_colunas_finais = processor.get_feature_names_out()
    df_final = pd.DataFrame(dados_processados, columns=nomes_colunas_finais)
except Exception as e:
    st.error(f"Erro no processamento: {e}")
    st.stop()

# --- MODO DEBUG (Expansível) ---
# with st.expander("🔍 Detalhes Técnicos (Dados enviados ao modelo)"):
    # st.write("Dados transformados (Devem estar escalonados entre -2 e 2 aprox.):")
    # st.dataframe(df_final)

# 5. Predição
if st.button("📊 Analisar Risco de Churn"):
    predicao = model.predict(df_final)[0]
    probabilidade = model.predict_proba(df_final)[0][1]
    
    st.divider()
    if predicao == 1:
        st.error(f"## ALTO RISCO DE CHURN: {probabilidade:.2%}")
        st.warning("Recomendação: Oferecer upgrade de plano ou desconto em contrato anual.")
    else:
        st.success(f"## BAIXO RISCO DE CHURN: {probabilidade:.2%}")
        st.info("Recomendação: Cliente estável. Manter régua de relacionamento padrão.")

st.sidebar.markdown("---")
st.sidebar.caption("Projeto Telecom X - Data Analytics")