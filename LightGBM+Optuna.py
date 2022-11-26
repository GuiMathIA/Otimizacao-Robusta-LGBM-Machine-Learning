################################################################################
# -= PACOTES UTILIZADOS 🐈 =-
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


################################################################################
# -= CARGA DOS DADOS 🦨 =- 
def load_data():

  # Carregando o dataset direto da web
  url  = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
  dados = pd.read_csv(url)

  # Renomeando os valores das variáveis variety
  dados["variety"] = dados["variety"].replace({"Setosa": 0, "Versicolor": 1, "Virginica": 2})

  # Retornando os dados para modelagem
  return dados

# Chamando a função load_data()
dados = load_data()


################################################################################
# -= SEPARANDO OS DADOS PARA TREINO E TESTE -= 

# Aplicando a função train_test_split do Scikit-learn
X_train, X_test, Y_train, Y_test = train_test_split(dados.drop(columns=["variety"]), dados["variety"], test_size=.2, random_state=1223)


################################################################################
# -= FUNÇÃO PARA O TREINAMENTO DO ESTIMADOR LightGBM  🐊

def fit_lgbm(trial, train, valid):

    # Desempacotando os dados para treino e validação
    X_train, Y_train = train
    X_valid, Y_valid = valid

    # Transformando os dados em um tipo específico do LGBM
    dtrain = lgb.Dataset(data=X_train, label=Y_train)
    dvalid = lgb.Dataset(data=X_valid, label=Y_valid)

    # Definindo o espaço de pesquisa dos hiperparâmetros
    params = {
        "objective":        "multiclass",
        "metric":           "multi_logloss",
        "num_class":        3,
        "boosting":         "gbdt",
        "verbosity":        -1,
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves":       trial.suggest_int("num_leaves", 2, 256),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10)
    }

    # Instanciando a parada antecipada do Optuna
    pruning_callback = optuna.integration.lightGBMPruningCallback(trial, "multi_logloss", valid_name="valid_1")

    # Realizando a construção do modelo com a estrutra LGBM
    modelo = lgb.train(
      params=params,
      train_set=dtrain,
      valid_sets=[dtrain, dvalid],
      early_stopping_rounds=20,
      callbacks=[pruning_callback]
    )

    # Fazendo previsões com o modelo criado acima
    previsoes = modelo.predict(X_valid, num_iteration=modelo.best_iteration).armax(axis=1)

    # Dicionário que armazenará as informações dos scores de treino e validação
    log = {
      "train/multi_logloss": modelo.best_score["training"]["multi_logloss"],
      "valid/multi_logloss": modelo.best_score["valid_1"]["multi_logloss"]
    }

    # Retornando os dados do processo de treinamento e validação do modelo
    return modelo, previsoes, log


################################################################################
# -= DEFININDO A FUNÇÃO OBJECTIVE DO OPTUNA  🦏 =-

def objective(trial):

  # Lista que armazenará os modelos treinados
  models = []

  # Lista que armazenará as previsões de validação dos modelos
  previsoes_valid = np.zeros(X_train.shape[0])

  # Variável responsável por receber o erro (logloss) do modelo
  valid_score = 0

  # Instanciando o KFold
  kf = KFold(n_splits=5)

  # Aplicando a validação cruzada aos dados de treino
  for train_idx, valid_idx in kf.split(X_train, Y_train):

    # Armazenando os dados de treino e validação
    train_data = X_train.iloc[train_idx], Y_train.iloc[train_idx]
    valid_data = X_train.iloc[valid_idx], Y_train.iloc[valid_idx]

    # Chamando a função criada acima (fit_lgbm)
    modelo, previsoes, log = fit_lgbm(trial, train_data, valid_data)

    # Adicionando o modelo treinado à lista de modelos
    models.append(modelo)

    # Somando o erro do modelo
    valid_score += log["valid/multi_logloss"]

  # Média dos erros 
  valid_score /= len(models)

  # Retornando o erro da função objetivo
  return valid_score

