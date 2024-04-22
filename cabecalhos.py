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

