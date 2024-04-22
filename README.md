# Dashboard para análise

Este projeto foi desenvolvido principalmente utilizando as seguintes bibliotecas em Python: Streamlit, Pandas, sklearn, plotly, numpy e statsmodels. Ele possui uma interface simples que permite a interação com o arquivo dataset.csv do usuário. No projeto, é possível analisar dados de três tipos:

## Séries Temporais

Séries Temporais são dados indexados no tempo. Os usuários podem inserir seus próprios dados e obter:

- Gráfico da série temporal.
- Histograma dos dados numéricos.
- Tabela de descrição dos dados.
- Gráfico da série temporal e de uma série secundária gerada pela média dos três últimos eventos.
- Um módulo de modelagem, onde é possível utilizar o modelo ETS para tentar prever as próximas ocorrências da série. Ele retorna o RMSE, a previsão do modelo ETS e a precisão usando a média dos três últimos eventos da série real.

## Classificação

No módulo de Classificação, é possível inserir dados de categorias ou classes e o usuário receberá:

- Um gráfico de barras mostrando as categorias, suas classes e suas respectivas quantidades.
- Um boxplot da variável alvo em relação à primeira variável secundária selecionada.
- Uma tabela com as variáveis selecionadas.
- Uma tabela sobre o modelo utilizado para classificação, que emprega um modelo de Florestas Aleatórias, utilizando a variável alvo e todas as secundárias.

O usuário recebe uma classificação baseada no modelo (este modelo não está otimizado e utiliza as configurações padrão do sklearn).

## Regressão

Neste módulo, o usuário pode modelar valores numéricos e obter:

- Um histograma da variável alvo.
- Uma matriz de correlação entre as variáveis selecionadas, incluindo ou não a coluna alvo.
- Uma tabela com informações sobre os dados.
- Um gráfico que pode ser um boxplot, se a primeira coluna secundária for categórica, ou um gráfico de dispersão, se for uma coluna numérica comum.

Também é possível criar um modelo de previsão, assim como na classificação, baseado em Florestas Aleatórias, porém este é otimizado usando uma busca em grade (GridSearch).

