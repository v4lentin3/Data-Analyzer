# Pacotes de manipulação de dados
import numpy as np
import pandas as pd
import streamlit as st

# Plotagem de graficos
import plotly.express as px
import plotly.graph_objects as go

# Pré-processamento
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Seleção de dados
from sklearn.model_selection import train_test_split
# Seleção de parametros
from sklearn.model_selection import GridSearchCV
# Modelos baseados em florestas
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier
# Modelo ETS
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Medidas de erros para regressão 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error

# Medidas de desempenho para classificação
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report


descricao = """ # Data Analyzer

O "Data Analyzer" é um projeto de dashboard de análise desenvolvido em Python com Streamlit, Pandas, sklearn, plotly, numpy e statsmodels. Ele oferece uma interface simples para interagir com dados do arquivo dataset.csv do usuário e realizar análises em três tipos principais de dados:

1. **Séries Temporais**: Permite visualizar gráficos e histogramas da série temporal, além de uma tabela descritiva dos dados. Também inclui um módulo de modelagem que utiliza o modelo ETS para prever futuras ocorrências da série.

2. **Classificação**: Permite analisar dados de categorias ou classes, mostrando gráficos de barras com a distribuição das categorias e um boxplot da variável alvo em relação a outras variáveis. Utiliza um modelo de Florestas Aleatórias para classificação.

3. **Regressão**: Oferece análises de valores numéricos, incluindo histogramas, matriz de correlação e gráficos de dispersão ou boxplot. Também utiliza um modelo otimizado de Florestas Aleatórias para predição.

Cada módulo fornece informações essenciais para análise exploratória e modelagem de dados, facilitando a compreensão e visualização dos dados inseridos pelo usuário.

Caso tenha interesse em saber mais sobre esse projeto e outros, você pode acessá-lo no GitHub:


[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/danttis)
"""

#Séries temporais

def is_numeric_column(column):
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False
    
def train_test(serie, test=0.2):
    cut_index = int(len(serie) * (1 - test))
    train = serie.iloc[:cut_index]
    test = serie.iloc[cut_index:]
    
    return train, test


def grid_ets(data):
    train_set, test_set = train_test(data)
    seasonal_types = ['add', 'mul']
    seasonal_periods = list(range(3, int(len(train_set) / 4)))
    error = 100000000
    best_seasonal_type = None
    best_seasonal_periods = None
    
    for seasonal_type in seasonal_types:
        for period in seasonal_periods:
            model = ETSModel(train_set, seasonal=seasonal_type, seasonal_periods=period)
            exp_model = model.fit()
            predictions = exp_model.forecast(steps=len(test_set))
            mse = mean_squared_error(test_set, predictions)
            
            if mse < error:
                best_seasonal_type = seasonal_type
                best_seasonal_periods = period
                error = mse
    
    return best_seasonal_type, best_seasonal_periods


def time_series_plot(train_frame, test_frame, predictions=None, label=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_frame['data'], y=train_frame['valor'], mode='lines+markers', name='Treino', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_frame['data'], y=test_frame['valor'], mode='lines+markers', name='Teste', line=dict(color='green')))
    if predictions is not None:
        fig.add_trace(go.Scatter(x=test_frame['data'], y=predictions, mode='lines+markers', name='Previsão', line=dict(color='red')))

    fig.update_layout(
        title=f'Previsão Série Temporal de {label} com ETS',
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(255, 255, 255, 0.5)'),
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )

    return fig


def time_series(data_ts, time_column, target_column):
    st.title("Análise de Série Temporal")
    coluna1, coluna2 = st.columns([3, 2])  
    coluna3, coluna4 = st.columns([2, 3]) 

    coluna1.plotly_chart(px.line(data_ts, x=time_column, y=target_column, title=f'Série Temporal de {target_column}', markers=True), use_container_width=True)
    coluna2.plotly_chart(px.histogram(data_ts, x=target_column, title=f'Histograma dos dados de {target_column}'), use_container_width=True)
    coluna3.table(data_ts[target_column].describe())
    
    if is_numeric_column(data_ts[target_column]):
        data_ts['Média das Últimas 3 Ocorrências'] = data_ts[target_column].rolling(3).mean() 

        fig = px.line(data_ts, x=time_column, y=[target_column, 'Média das Últimas 3 Ocorrências'], 
                      title='Série Temporal e Média Móvel', 
                      color_discrete_map={target_column: 'blue', 'Média das Últimas 3 Ocorrências': 'red'})

        coluna4.plotly_chart(fig)

        show_modeling_options = st.sidebar.checkbox("Modelar")

        if show_modeling_options:
            #coluna5 = st.columns(1)
            test_percentage = 0.2
            
            data_model = pd.DataFrame({'data': data_ts[time_column], 'valor': data_ts[target_column]})
            train, test = train_test(data_model, test=test_percentage)
            
            seasonal, seasonal_periods = grid_ets(train['valor'])
            model = ETSModel(train['valor'], seasonal=seasonal, seasonal_periods=seasonal_periods)
            exp_model = model.fit()
            predictions = exp_model.forecast(steps=len(test)+1)
            predictions = list(predictions)
            novo_valor = predictions[-1]
            del predictions[-1]
            mse = mean_squared_error(test['valor'], predictions)
            print("Erro quadrático médio (MSE):", mse)
            grafico = time_series_plot(train, test, predictions, label=target_column)
            test_list = list(test['valor'])
            erro = mean_squared_error(test_list, predictions)
            st.plotly_chart(grafico)
            st.text(f'MSE: {erro}')
            st.text(f'Pelo ETS a próxima ocorrência será: {novo_valor}')
            st.text(f'Pelos 3 últimos ocorridos, a próxima ocorrência será: {np.mean(test_list[-3:])}')
    else:
        coluna4.warning("Selecione uma coluna numérica!")


def pipeline_full(data): # O pipeline é o mesmo usando na classificação e regressão, pois o que ele faz é adicionar a média nos dados numericos e a moda em dados não numericos
    numeric_df = data.select_dtypes(include=['number'])
    categorical_df = data.select_dtypes(exclude=['number'])

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median", add_indicator=False)),
        ('std_scaler', StandardScaler(with_mean=False))
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent", add_indicator=False)),
        ('ohe', OneHotEncoder()),
        ('std_scaler', StandardScaler(with_mean=False))
    ]) 

    full_pipeline = ColumnTransformer([
        ("numeric_transform", numeric_pipeline, numeric_df.columns),
        ("categorical_transform", categorical_pipeline, categorical_df.columns)
    ]) 
    
    return full_pipeline

# Classificação 

def classificacao(data_frame, target_variable, selected_options):
    coluna1, coluna2 = st.columns(2)
    quantidade = data_frame[target_variable].value_counts()
    cores = px.colors.qualitative.Plotly[:len(quantidade)]
    figura = go.Figure(data=[go.Bar(x=quantidade.index, y=quantidade.values, marker_color=cores)])

    figura.update_layout(
        title=f'Categorias de {target_variable}',
        xaxis_title=target_variable,
        yaxis_title='Quantidade'
    )

    coluna1.plotly_chart(figura)

    if len(selected_options) > 0:  
        def analise_variaveis_quali_quanti(dados, variavel1, variavel2):
            figura = px.box(dados, x=variavel1, y=variavel2, title=f"{variavel1} por {variavel2}")
            figura.update_layout(
                xaxis_title=variavel1,
                yaxis_title=variavel2,
                title_font_size=14
            )

            # Exibindo o gráfico
            coluna2.plotly_chart(figura, use_container_width=True)
        
        analise_variaveis_quali_quanti(data_frame, target_variable, selected_options[0])

        data_frame = data_frame[[target_variable] + selected_options]
        st.text('Variáveis:')
        st.table(data_frame.head())
        data_frame.dropna(inplace=True)     
        label_encoder = LabelEncoder()
        is_quantitative = False    
        if not is_numeric_column(data_frame[target_variable]):
            label_encoder.fit(data_frame[target_variable])
            data_frame[target_variable] = label_encoder.fit_transform(data_frame[target_variable])
            is_quantitative = True

        treino, teste = train_test_split(data_frame, test_size=0.2)
        teste_y = teste[target_variable]
        pipe = pipeline_full(treino[selected_options])
        df_treino = pd.DataFrame(pipe.fit_transform(treino), columns=pipe.get_feature_names_out())
        
        X = df_treino
        Y = treino[target_variable]
        teste2 = pd.DataFrame(pipe.fit_transform(teste), columns=pipe.get_feature_names_out())
        modelo_rf = RandomForestClassifier()
        modelo_rf.fit(X, Y)
        previsao = modelo_rf.predict(teste2)
        #acuracia = accuracy_score(teste_y, previsao)
    
        if is_quantitative:
            classes = dict(zip(list(label_encoder.classes_), label_encoder.transform(label_encoder.classes_)))
            st.table(classes)

        report = classification_report(teste[target_variable], previsao, output_dict=True) # é preferivél a a matrizes de confusão normais já que esse pacote trás mais dados 

        metrics_data = []
        for class_label, metrics in report.items():
            if class_label.isdigit():  
                metrics_data.append({
                    'class': int(class_label),
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                })
        metrics_df = pd.DataFrame(metrics_data)
        st.text('Acurácia do modelo:')
        st.table(metrics_df)

        texto = st.chat_input('Digite os valores das colunas independentes e preveja o resultado, separe por vírgula:')
        if texto:
            #valores = [valor.strip() for valor in texto.split(',')]
            valores = [valor.strip() for valor in texto.split(',')]
            if len(valores) >= len(selected_options):
                novo_evento = dict(zip(selected_options, valores))
                dados_novos = pd.DataFrame([novo_evento])
                dados_novos = pd.DataFrame(pipe.fit_transform(dados_novos), columns=pipe.get_feature_names_out())
                colunas_faltantes = [item for item in list(df_treino.columns) if item not in list(dados_novos.columns)]    
                dados_novos[colunas_faltantes] = 0
                dados_novos = dados_novos[df_treino.columns]
                previsao = modelo_rf.predict(dados_novos)
                st.text(f"Para esses dados é esperado que seja da classe: {previsao}")
            else:
                st.warning(f'Deve digitar ao menos {len(selected_options)} valores')

#Regressão


def regressao(data_frame, target_variable, selected_options):
    coluna1, coluna2 = st.columns(2)
    coluna1.plotly_chart(px.histogram(data_frame, x=target_variable, title=f'Histograma dos dados de {target_variable}'), use_container_width=True)
    coluna1.text("Descrição das Variáveis")
    coluna1.table(data_frame.describe())
    correlation_matrix = data_frame[[target_variable]+selected_options].corr()

    features = correlation_matrix.columns.tolist()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=features,
        y=features,
        colorscale='Viridis',  
        colorbar=dict(title='Correlação')
    ))

    fig.update_layout(
        title='Matriz de Correlação',
        height=500,
        xaxis_title='Features',
        yaxis_title='Features'
    )

    coluna2.plotly_chart(fig, use_container_width=True)    

    if len(selected_options) > 0:
        def plotar_grafico_dispersao(dados, variavel1, variavel2): 
            if len(dados[variavel1].unique()) < 10:
                figura = px.box(dados, x=variavel1, y=variavel2, title=f"{variavel1} por {variavel2}")
            else:
                figura = px.scatter(dados, x=variavel1, y=variavel2, title=f"{variavel1} por {variavel2}")
            figura.update_layout(
                xaxis_title=variavel1,
                yaxis_title=variavel2,
                title_font_size=14
            )
            coluna2.plotly_chart(figura, use_container_width=True)
        
        plotar_grafico_dispersao(data_frame, selected_options[0], target_variable)

        modelar = st.sidebar.selectbox("Modelar: ", ["Não", "Sim"]) 
        if modelar == 'Sim':
            treino, teste = train_test_split(data_frame, test_size=0.2, random_state=17)
            X, Y = treino[selected_options], treino[target_variable]
            pipe = pipeline_full(X)
            treino = pd.DataFrame(pipe.fit_transform(X), columns=pipe.get_feature_names_out())   

            randomForestParamsSearch = {'n_estimators': [40, 50, 100],
                                        'max_depth': [None, 5, treino.shape[1]],  
                                        'max_features': [6, 8, treino.shape[1]], 
                                        'min_samples_split': [2, 10, 12]} 
            forest_reg2 = RandomForestRegressor(random_state=0, bootstrap=False) 
            grid_search = GridSearchCV(forest_reg2, randomForestParamsSearch, cv=3, scoring='neg_mean_squared_error', 
                                    return_train_score=True, verbose=True) 
            grid_search.fit(treino, Y.values.ravel())
            forest_reg2 = grid_search.best_estimator_

            
            
            y_true = teste[target_variable]
            teste = pd.DataFrame(pipe.fit_transform(teste[selected_options]), columns=pipe.get_feature_names_out())

            def mostrar_desempenho_modelo(modelos): 
                colunas = ['Modelo', 'R2', 'MAE', 'RMSE', 'RMSLE']
                tabela_modelos = pd.DataFrame(columns=colunas)
                for nome, modelo in modelos.items():
                    y_predito = modelo.predict(teste)
                    y_predito_log = np.log1p(y_predito)
                    r2 = r2_score(y_true, y_predito)
                    mae = mean_absolute_error(y_true, y_predito)
                    rmse = np.sqrt(mean_squared_error(y_true, y_predito))
                    rmsle = np.sqrt(mean_squared_log_error(y_true, y_predito_log))
                    nova_linha = pd.DataFrame([[nome, r2, mae, rmse, rmsle]], columns=colunas)
                    tabela_modelos = pd.concat([tabela_modelos, nova_linha], axis=0)
                tabela_modelos = tabela_modelos.sort_values(by='R2', ascending=False)                
                return tabela_modelos
                        
            modelos_individuais = [forest_reg2]
            nomes_modelos = ['Random Forest']
            dicionario_modelos = {nome: modelo for nome, modelo in zip(nomes_modelos, modelos_individuais)}
            tabela_modelos = mostrar_desempenho_modelo(dicionario_modelos)
            st.table(tabela_modelos)
            texto = st.chat_input('Digite os valores das colunas independentes e preveja o resultado, separados por vírgula:')
            if texto:
                valores = [valor.strip() for valor in texto.split(',')]
                if len(valores) >= len(selected_options):
                    novo_evento = dict(zip(selected_options, valores))
                    dados_novos = pd.DataFrame([novo_evento])
                    dados_novos = pd.DataFrame(pipe.fit_transform(dados_novos), columns=pipe.get_feature_names_out())
                    colunas_faltantes = [item for item in list(treino.columns) if item not in list(dados_novos.columns)]    
                    dados_novos[colunas_faltantes] = 0
                    dados_novos = dados_novos[treino.columns]
                    previsao = forest_reg2.predict(dados_novos)
                    st.text(f"O valor esperado para {target_variable} é: {previsao}")
                else:
                    st.warning(f'Deve digitar pelo menos {len(selected_options)} valores')


descricao = """ # Data Analyzer

O "Data Analyzer" é um projeto de dashboard de análise desenvolvido em Python com Streamlit, Pandas, sklearn, plotly, numpy e statsmodels. Ele oferece uma interface simples para interagir com dados do arquivo dataset.csv do usuário e realizar análises em três tipos principais de dados:

1. **Séries Temporais**: Permite visualizar gráficos e histogramas da série temporal, além de uma tabela descritiva dos dados. Também inclui um módulo de modelagem que utiliza o modelo ETS para prever futuras ocorrências da série.

2. **Classificação**: Permite analisar dados de categorias ou classes, mostrando gráficos de barras com a distribuição das categorias e um boxplot da variável alvo em relação a outras variáveis. Utiliza um modelo de Florestas Aleatórias para classificação.

3. **Regressão**: Oferece análises de valores numéricos, incluindo histogramas, matriz de correlação e gráficos de dispersão ou boxplot. Também utiliza um modelo otimizado de Florestas Aleatórias para predição.

Cada módulo fornece informações essenciais para análise exploratória e modelagem de dados, facilitando a compreensão e visualização dos dados inseridos pelo usuário.

Caso tenha interesse em saber mais sobre esse projeto e outros, você pode acessá-lo no GitHub:


[![Github](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/danttis)
"""

#Séries temporais

def is_numeric_column(column):
    try:
        pd.to_numeric(column)
        return True
    except ValueError:
        return False
    
def train_test(serie, test=0.2):
    cut_index = int(len(serie) * (1 - test))
    train = serie.iloc[:cut_index]
    test = serie.iloc[cut_index:]
    
    return train, test


def grid_ets(data):
    train_set, test_set = train_test(data)
    seasonal_types = ['add', 'mul']
    seasonal_periods = list(range(3, int(len(train_set) / 4)))
    error = 100000000
    best_seasonal_type = None
    best_seasonal_periods = None
    
    for seasonal_type in seasonal_types:
        for period in seasonal_periods:
            model = ETSModel(train_set, seasonal=seasonal_type, seasonal_periods=period)
            exp_model = model.fit()
            predictions = exp_model.forecast(steps=len(test_set))
            mse = mean_squared_error(test_set, predictions)
            
            if mse < error:
                best_seasonal_type = seasonal_type
                best_seasonal_periods = period
                error = mse
    
    return best_seasonal_type, best_seasonal_periods


def time_series_plot(train_frame, test_frame, predictions=None, label=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_frame['data'], y=train_frame['valor'], mode='lines+markers', name='Treino', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_frame['data'], y=test_frame['valor'], mode='lines+markers', name='Teste', line=dict(color='green')))
    if predictions is not None:
        fig.add_trace(go.Scatter(x=test_frame['data'], y=predictions, mode='lines+markers', name='Previsão', line=dict(color='red')))

    fig.update_layout(
        title=f'Previsão Série Temporal de {label} com ETS',
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(255, 255, 255, 0.5)'),
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )

    return fig


def time_series(data_ts, time_column, target_column):
    st.title("Análise de Série Temporal")
    coluna1, coluna2 = st.columns([3, 2])  
    coluna3, coluna4 = st.columns([2, 3]) 

    coluna1.plotly_chart(px.line(data_ts, x=time_column, y=target_column, title=f'Série Temporal de {target_column}', markers=True), use_container_width=True)
    coluna2.plotly_chart(px.histogram(data_ts, x=target_column, title=f'Histograma dos dados de {target_column}'), use_container_width=True)
    coluna3.table(data_ts[target_column].describe())
    
    if is_numeric_column(data_ts[target_column]):
        data_ts['Média das Últimas 3 Ocorrências'] = data_ts[target_column].rolling(3).mean() 

        fig = px.line(data_ts, x=time_column, y=[target_column, 'Média das Últimas 3 Ocorrências'], 
                      title='Série Temporal e Média Móvel', 
                      color_discrete_map={target_column: 'blue', 'Média das Últimas 3 Ocorrências': 'red'})

        coluna4.plotly_chart(fig)

        show_modeling_options = st.sidebar.checkbox("Modelar")

        if show_modeling_options:
            #coluna5 = st.columns(1)
            test_percentage = 0.2
            
            data_model = pd.DataFrame({'data': data_ts[time_column], 'valor': data_ts[target_column]})
            train, test = train_test(data_model, test=test_percentage)
            
            seasonal, seasonal_periods = grid_ets(train['valor'])
            model = ETSModel(train['valor'], seasonal=seasonal, seasonal_periods=seasonal_periods)
            exp_model = model.fit()
            predictions = exp_model.forecast(steps=len(test)+1)
            predictions = list(predictions)
            novo_valor = predictions[-1]
            del predictions[-1]
            mse = mean_squared_error(test['valor'], predictions)
            print("Erro quadrático médio (MSE):", mse)
            grafico = time_series_plot(train, test, predictions, label=target_column)
            test_list = list(test['valor'])
            erro = mean_squared_error(test_list, predictions)
            st.plotly_chart(grafico)
            st.text(f'MSE: {erro}')
            st.text(f'Pelo ETS a próxima ocorrência será: {novo_valor}')
            st.text(f'Pelos 3 últimos ocorridos, a próxima ocorrência será: {np.mean(test_list[-3:])}')
    else:
        coluna4.warning("Selecione uma coluna numérica!")


def pipeline_full(data): # O pipeline é o mesmo usando na classificação e regressão, pois o que ele faz é adicionar a média nos dados numericos e a moda em dados não numericos
    numeric_df = data.select_dtypes(include=['number'])
    categorical_df = data.select_dtypes(exclude=['number'])

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median", add_indicator=False)),
        ('std_scaler', StandardScaler(with_mean=False))
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent", add_indicator=False)),
        ('ohe', OneHotEncoder()),
        ('std_scaler', StandardScaler(with_mean=False))
    ]) 

    full_pipeline = ColumnTransformer([
        ("numeric_transform", numeric_pipeline, numeric_df.columns),
        ("categorical_transform", categorical_pipeline, categorical_df.columns)
    ]) 
    
    return full_pipeline

# Classificação 

def classificacao(data_frame, target_variable, selected_options):
    coluna1, coluna2 = st.columns(2)
    quantidade = data_frame[target_variable].value_counts()
    cores = px.colors.qualitative.Plotly[:len(quantidade)]
    figura = go.Figure(data=[go.Bar(x=quantidade.index, y=quantidade.values, marker_color=cores)])

    figura.update_layout(
        title=f'Categorias de {target_variable}',
        xaxis_title=target_variable,
        yaxis_title='Quantidade'
    )

    coluna1.plotly_chart(figura)

    if len(selected_options) > 0:  
        def analise_variaveis_quali_quanti(dados, variavel1, variavel2):
            figura = px.box(dados, x=variavel1, y=variavel2, title=f"{variavel1} por {variavel2}")
            figura.update_layout(
                xaxis_title=variavel1,
                yaxis_title=variavel2,
                title_font_size=14
            )

            # Exibindo o gráfico
            coluna2.plotly_chart(figura, use_container_width=True)
        
        analise_variaveis_quali_quanti(data_frame, target_variable, selected_options[0])

        data_frame = data_frame[[target_variable] + selected_options]
        st.text('Variáveis:')
        st.table(data_frame.head())
        data_frame.dropna(inplace=True)     
        label_encoder = LabelEncoder()
        is_quantitative = False    
        if not is_numeric_column(data_frame[target_variable]):
            label_encoder.fit(data_frame[target_variable])
            data_frame[target_variable] = label_encoder.fit_transform(data_frame[target_variable])
            is_quantitative = True

        treino, teste = train_test_split(data_frame, test_size=0.2)
        teste_y = teste[target_variable]
        pipe = pipeline_full(treino[selected_options])
        df_treino = pd.DataFrame(pipe.fit_transform(treino), columns=pipe.get_feature_names_out())
        
        X = df_treino
        Y = treino[target_variable]
        teste2 = pd.DataFrame(pipe.fit_transform(teste), columns=pipe.get_feature_names_out())
        modelo_rf = RandomForestClassifier()
        modelo_rf.fit(X, Y)
        previsao = modelo_rf.predict(teste2)
        #acuracia = accuracy_score(teste_y, previsao)
    
        if is_quantitative:
            classes = dict(zip(list(label_encoder.classes_), label_encoder.transform(label_encoder.classes_)))
            st.table(classes)

        report = classification_report(teste[target_variable], previsao, output_dict=True) # é preferivél a a matrizes de confusão normais já que esse pacote trás mais dados 

        metrics_data = []
        for class_label, metrics in report.items():
            if class_label.isdigit():  
                metrics_data.append({
                    'class': int(class_label),
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                })
        metrics_df = pd.DataFrame(metrics_data)
        st.text('Acurácia do modelo:')
        st.table(metrics_df)

        texto = st.chat_input('Digite os valores das colunas independentes e preveja o resultado, separe por vírgula:')
        if texto:
            #valores = [valor.strip() for valor in texto.split(',')]
            valores = [valor.strip() for valor in texto.split(',')]
            if len(valores) >= len(selected_options):
                novo_evento = dict(zip(selected_options, valores))
                dados_novos = pd.DataFrame([novo_evento])
                dados_novos = pd.DataFrame(pipe.fit_transform(dados_novos), columns=pipe.get_feature_names_out())
                colunas_faltantes = [item for item in list(df_treino.columns) if item not in list(dados_novos.columns)]    
                dados_novos[colunas_faltantes] = 0
                dados_novos = dados_novos[df_treino.columns]
                previsao = modelo_rf.predict(dados_novos)
                st.text(f"Para esses dados é esperado que seja da classe: {previsao}")
            else:
                st.warning(f'Deve digitar ao menos {len(selected_options)} valores')

#Regressão


def regressao(data_frame, target_variable, selected_options):
    coluna1, coluna2 = st.columns(2)
    coluna1.plotly_chart(px.histogram(data_frame, x=target_variable, title=f'Histograma dos dados de {target_variable}'), use_container_width=True)
    coluna1.text("Descrição das Variáveis")
    coluna1.table(data_frame.describe())
    correlation_matrix = data_frame[[target_variable]+selected_options].corr()

    features = correlation_matrix.columns.tolist()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=features,
        y=features,
        colorscale='Viridis',  
        colorbar=dict(title='Correlação')
    ))

    fig.update_layout(
        title='Matriz de Correlação',
        height=500,
        xaxis_title='Features',
        yaxis_title='Features'
    )

    coluna2.plotly_chart(fig, use_container_width=True)    

    if len(selected_options) > 0:
        def plotar_grafico_dispersao(dados, variavel1, variavel2): 
            if len(dados[variavel1].unique()) < 10:
                figura = px.box(dados, x=variavel1, y=variavel2, title=f"{variavel1} por {variavel2}")
            else:
                figura = px.scatter(dados, x=variavel1, y=variavel2, title=f"{variavel1} por {variavel2}")
            figura.update_layout(
                xaxis_title=variavel1,
                yaxis_title=variavel2,
                title_font_size=14
            )
            coluna2.plotly_chart(figura, use_container_width=True)
        
        plotar_grafico_dispersao(data_frame, selected_options[0], target_variable)

        modelar = st.sidebar.selectbox("Modelar: ", ["Não", "Sim"]) 
        if modelar == 'Sim':
            treino, teste = train_test_split(data_frame, test_size=0.2, random_state=17)
            X, Y = treino[selected_options], treino[target_variable]
            pipe = pipeline_full(X)
            treino = pd.DataFrame(pipe.fit_transform(X), columns=pipe.get_feature_names_out())   

            randomForestParamsSearch = {'n_estimators': [40, 50, 100],
                                        'max_depth': [None, 5, treino.shape[1]],  
                                        'max_features': [6, 8, treino.shape[1]], 
                                        'min_samples_split': [2, 10, 12]} 
            forest_reg2 = RandomForestRegressor(random_state=0, bootstrap=False) 
            grid_search = GridSearchCV(forest_reg2, randomForestParamsSearch, cv=3, scoring='neg_mean_squared_error', 
                                    return_train_score=True, verbose=True) 
            grid_search.fit(treino, Y.values.ravel())
            forest_reg2 = grid_search.best_estimator_

            
            
            y_true = teste[target_variable]
            teste = pd.DataFrame(pipe.fit_transform(teste[selected_options]), columns=pipe.get_feature_names_out())

            def mostrar_desempenho_modelo(modelos): 
                colunas = ['Modelo', 'R2', 'MAE', 'RMSE', 'RMSLE']
                tabela_modelos = pd.DataFrame(columns=colunas)
                for nome, modelo in modelos.items():
                    y_predito = modelo.predict(teste)
                    y_predito_log = np.log1p(y_predito)
                    r2 = r2_score(y_true, y_predito)
                    mae = mean_absolute_error(y_true, y_predito)
                    rmse = np.sqrt(mean_squared_error(y_true, y_predito))
                    rmsle = np.sqrt(mean_squared_log_error(y_true, y_predito_log))
                    nova_linha = pd.DataFrame([[nome, r2, mae, rmse, rmsle]], columns=colunas)
                    tabela_modelos = pd.concat([tabela_modelos, nova_linha], axis=0)
                tabela_modelos = tabela_modelos.sort_values(by='R2', ascending=False)                
                return tabela_modelos
                        
            modelos_individuais = [forest_reg2]
            nomes_modelos = ['Random Forest']
            dicionario_modelos = {nome: modelo for nome, modelo in zip(nomes_modelos, modelos_individuais)}
            tabela_modelos = mostrar_desempenho_modelo(dicionario_modelos)
            st.table(tabela_modelos)
            texto = st.chat_input('Digite os valores das colunas independentes e preveja o resultado, separados por vírgula:')
            if texto:
                valores = [valor.strip() for valor in texto.split(',')]
                if len(valores) >= len(selected_options):
                    novo_evento = dict(zip(selected_options, valores))
                    dados_novos = pd.DataFrame([novo_evento])
                    dados_novos = pd.DataFrame(pipe.fit_transform(dados_novos), columns=pipe.get_feature_names_out())
                    colunas_faltantes = [item for item in list(treino.columns) if item not in list(dados_novos.columns)]    
                    dados_novos[colunas_faltantes] = 0
                    dados_novos = dados_novos[treino.columns]
                    previsao = forest_reg2.predict(dados_novos)
                    st.text(f"O valor esperado para {target_variable} é: {previsao}")
                else:
                    st.warning(f'Deve digitar pelo menos {len(selected_options)} valores')


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
