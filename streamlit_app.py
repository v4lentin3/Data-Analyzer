from cabecalhos import *
from funcoes import time_series, classificacao, regressao, descricao

tipo_descricoes = {
    'Série Temporal': 'Dados que representam observações ao longo do tempo.',
    'Classificação': 'Dados usados para prever uma categoria ou classe.',
    'Regressão': 'Dados usados para prever um valor numérico.'
}

st.set_page_config(
    page_title="Data Analyzer",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

data_file = st.sidebar.file_uploader("Carregar arquivo CSV")

show_additional_options = st.sidebar.checkbox("Opções Avançadas")
separator, decimal_point = ',', '.'

if show_additional_options:
    separator = st.sidebar.selectbox("Separador", [',', ';', '.', '|', ' '])
    decimal_point = st.sidebar.selectbox("Decimal", ['.', ','])

if data_file is not None:
    df = pd.read_csv(data_file, sep=separator, decimal=decimal_point)
    
    tipo = st.sidebar.selectbox("Os dados que deseja analisar são de qual tipo?", ['']+list(tipo_descricoes.keys()))
    if tipo == '':
        st.markdown(descricao)

    if tipo == 'Série Temporal':
        st.sidebar.write(f'Descrição: {tipo_descricoes[tipo]}')
        x_axis = st.sidebar.selectbox("Selecione o eixo X (Datas):", df.columns)
        y_axis = st.sidebar.selectbox("Selecione o eixo Y (Valores):", df.drop(x_axis, axis=1).columns)    
        time_series(df, x_axis, y_axis)

    elif tipo == 'Classificação':
        st.sidebar.write(f'Descrição: {tipo_descricoes[tipo]}')
        target_column = st.sidebar.selectbox("Selecione a coluna alvo:", df.columns)
        options = df.drop(target_column, axis=1).columns
        selected_options = st.sidebar.multiselect('Selecione as variáveis independentes:', options)
        classificacao(df, target_column, selected_options)

    elif tipo == 'Regressão':
        st.sidebar.write(f'Descrição: {tipo_descricoes[tipo]}')
        target_column = st.sidebar.selectbox("Selecione a coluna alvo:", df.columns)
        options = df.drop(target_column, axis=1).columns
        selected_options = st.sidebar.multiselect('Selecione as variáveis independentes:', options)
        regressao(df, target_column, selected_options)

else:
    
    st.markdown(descricao)
    st.warning("Por favor, suba um arquivo CSV.")
