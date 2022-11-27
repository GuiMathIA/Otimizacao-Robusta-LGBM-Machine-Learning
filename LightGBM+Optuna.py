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

  # Desempacotando os dados para treino e validação do modelo
  X_train, Y_train = train
  X_valid, Y_valid = valid

  # Transformando os dados em um tipo específico do LGBM
  dtrain = lgb.Dataset(data=X_train, label=Y_train)
  dvalid = lgb.Dataset(data=X_valid, label=Y_valid)

  # Definindo o espaço de pesquisa dos hiperparâmetros
  params = {
      "objective":        "multiclass",
      "metric":           "multi_logloss",
      "boosting":         "gbdt",
      "seed":             1223,
      "verbosity":        -1,
      "num_class":        3,
      "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1),
      "num_leaves":       trial.suggest_int("num_leaves", 2, 256),
      "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
      "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
      "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
      "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
      "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10)
  }

  # Instanciando a poda integrada do Optuna
  pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss", valid_name="valid_1")

  # Realizando o treinamento com o algoritmo LGBM
  modelo = lgb.train(
      params=params,
      train_set=dtrain,
      valid_sets=[dtrain, dvalid],
      early_stopping_rounds=20,
      callbacks=[pruning_callback]
  )

  # Salvando os resultados de treinamento e validação do modelo
  log = {
      "train/multi_logloss": modelo.best_score["training"]["multi_logloss"],
      "valid/multi_logloss": modelo.best_score["valid_1"]["multi_logloss"]
  }

  # Retornando todas as informações do processo de pesquisa dos hiperparâmetros
  return log

################################################################################
# -= DEFININDO A FUNÇÃO OBJECTIVE DO OPTUNA  🦏 =-

# Criando função objetivo do Optuna
def objective(trial):

  # Definindo o número de K-Folds para a validação cruzada
  kf = KFold(n_splits=5, shuffle=True, random_state=1223)

  # Definindo a variável que armazenará o score de validação dos modelos
  valid_score = 0

  # Aplicando uma validação cruzada
  for train_idx, valid_idx in kf.split(X_train, Y_train):

    # Empacotando os dados para treino e validação
    train_data = X_train.iloc[train_idx], Y_train.iloc[train_idx]
    valid_data = X_train.iloc[valid_idx], Y_train.iloc[valid_idx]

    # Chamando a função para treinar o modelo com o algoritmo LGBM
    log = fit_lgbm(trial, train_data, valid_data)

    # Salvando o erro do modelo
    valid_score += log["valid/multi_logloss"]

  # Retornando a métrica de otimização para a função objective
  return valid_score / 5


################################################################################
# -= REALIZANDO A PESQUISA DE HIPERPARÂMETROS 🦅 =-

# Definindo o tempo de pesquisa de hiperparâmetros
tempo = 60 * 60 * 0.1  # 6 minutos
print(f'Tempo de Pesquisa: {tempo / 60} min')

# Criando um objeto de estudo do Optuna
study = optuna.create_study(pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=4, min_early_stopping_rate=1))
study.optimize(objective, timeout=tempo)