# statstech

Biblioteca de aprendizado de máquina com NumPy.
Projeto final de Aprendizado de Máquina I (IMPA Tech, 2026.2)

## Instalação

Requer Python 3.11 ou mais recente.

```bash
pip install git+https://github.com/ml1-impatech/statstech.git
```

<!-- Quando publicar no PyPI trocar o comando acima por: 
pip install statstech
-->

## Exemplo
 
```python
import numpy as np
 
from statstech.model_selection import train_test_split
from statstech.neural_network import MLPRegressor
from statstech.preprocessing import StandardScaler
 
rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(300, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
 
# O scaler aprende média e desvio só no treino.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
 
model = MLPRegressor(hidden_layer_sizes=(32,), learning_rate=1e-2, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # R² no teste
```

 Há um exemplo para cada módulo na pasta [`examples/`](examples).

## Módulos e métodos

<!-- Manter esta tabela igual ao que está na main. Tirar o que não entrar. -->

| Módulo | Métodos |
|---|---|
| `linear_model` | Mínimos quadrados, Ridge, Lasso, regressão logística |
| `neighbors` | KNN |
| `discriminant_analysis` | LDA, QDA |
| `tree` | Árvores de decisão (regressão e classificação) |
| `neural_network` | Redes neurais (regressão) |
| `decomposition` | PCA |
| `cluster` | K-means |
| `model_selection` | Divisão treino/teste, k-fold, LOOCV, bootstrap, busca de hiperparâmetros |
| `metrics` | Métricas de regressão e de classificação |
| `preprocessing` | Padronização, codificação de categorias, imputação de valores faltantes, features polinomiais (para regressão polinomial) |
| `pipeline` | Encadeamento de pré-processamento e modelo, reajustado em cada fold da validação cruzada |

## Testes

```bash
pytest
```

## Referências

James, G., Witten, D., Hastie, T., Tibshirani, R. e Taylor, J. *An Introduction
to Statistical Learning with Applications in Python*. Springer, 2023.

## Autores

| Nome | GitHub |
|---|---|
| Daniela Moraes | [daniela-mmoraes](https://github.com/daniela-mmoraes) |
| Dulce Costa | [dulce-costa](https://github.com/dulce-costa) |
| Julia Fonseca | [JuliaQuiuquiF](https://github.com/JuliaQuiuquiF) |
| Lucas Silva | [lucas-impatech](https://github.com/lucas-impatech) |
| Natália Sousa | [NataliaBrandaodeSousa](https://github.com/NataliaBrandaodeSousa) |
| Manuela Ronconi | [manuronconi](https://github.com/manuronconi) |
| Maria Clara Fernandes | [mclaracarvalhoo](https://github.com/mclaracarvalhoo) |
| Maria Izabelle Silva | [izabelle02](https://github.com/izabelle02) |

## Uso de IA 

Seguindo as políticas internas da instituição, o Claude (Anthropic) foi utilizado como ferramenta de consulta para explicações de conceito e sintaxe, revisão de código e rascunhos de docstrings. 

## Licença

MIT. Veja o arquivo `LICENSE`.
