# Decisões 

Este documento registra as decisões tomadas na construção da biblioteca `statstech`.

## 1. Organização do repositório

- Inspirado na biblioteca [CB2325NumericaG8](https://github.com/lazyluca/CB2325NumericaG8) construída pelos alunos do IMPA-Tech na disciplina de **Programação 2 2025**
- **`src/`**: o código fica em `src/statstech/`, uma pasta por módulo
  (`linear_model`, `tree`, `neural_network`, `metrics` etc.).
- **`tests/`** para cada módulo.
- **`examples/`** com um exemplo simples de uso para cada módulo.
- **`docs/`** guarda este documento.

## 2. Classe ou função

- **Classe** quando o objeto guarda estado aprendido: modelos e
  transformadores.
- **Função** quando só recebe entrada e devolve saída, sem estado: métricas,
  funções de validação e de reamostragem.

## 3. Código compartilhado

Funcionalidades usadas por vários módulos ficam localizadas em um mesmo arquivo para evitar repetições de funções.

- **`base.py`**
- **`utils/validation.py`** (Todos os modelos validam entradas do mesmo jeito e levantam as mesmas mensagens de erro).


## 3. Pré-processamento 

- O modelo **não padroniza** os dados por dentro. Quem usa padroniza antes,
com o `StandardScaler` do módulo `preprocessing`.
- Motivos: evita vazamento de dados na validação cruzada, evita repetir
código em cada modelo e mantém cada módulo com uma responsabilidade.

## 4. Estilo de código

- **PEP 8**.
- **Docstrings no padrão NumPy** (`Parameters`, `Returns`, `Raises`) em português.

## 5. Fluxo de trabalho no Git

- A `main` é **protegida**.
  Mudança por Pull Request.
- Uma branch por tarefa.
- Problemas e propostas de mudança são registrados como *issues*.

## 6. Instalação

- Instalável por `pip install git+https://github.com/ml1-impatech/statstech.git`.

## 7. Análise discriminante (LDA e QDA)

- **Regularização:** `LDA` e `QDA` utilizam `reg_param=1e-6` como valor padrão.
- **Validação de `reg_param`:** ambos os classificadores rejeitam valores negativos, não finitos e booleanos.
- **Priors personalizadas:** `QDA` aceita o parâmetro `priors`, assim como `LDA`. Quando `priors=None`, as probabilidades a priori são estimadas pelas frequências das classes no conjunto de treinamento. 

