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

- **`base.py`**: classe base dos modelos, mixins e `clone`.
- **`utils/validation.py`**: todos os modelos validam entradas do mesmo jeito e
  levantam as mesmas mensagens de erro (`check_X_y`, `check_array`,
  `check_is_fitted`, `check_n_features`).

## 4. Validação de hiperparâmetros
 
- O `__init__` só guarda os hiperparâmetros, sem validar nada.
- A validação fica em um método `_check_params()`, chamado no início do `fit`.
  Assim, um valor passado por `set_params` também é validado.

## 5. Pré-processamento 

- O modelo **não padroniza** os dados por dentro. Quem usa padroniza antes,
com o `StandardScaler` do módulo `preprocessing`.
- Motivos: evita vazamento de dados na validação cruzada, evita repetir
código em cada modelo e mantém cada módulo com uma responsabilidade.

## 6. Estilo de código

- **PEP 8**, verificado com `ruff check` e `ruff format`.
- **Código em inglês** (nomes de classes, funções e variáveis) e **docstrings
  e comentários em português**.
- **Docstrings no padrão NumPy** (`Parameters`, `Returns`, `Raises`).
- **Type hints** nos parâmetros e no retorno de todas as funções e métodos.

## 7. Fluxo de trabalho no Git

- A `main` é **protegida**.
  Mudança por Pull Request.
- Uma branch por tarefa.
- Problemas e propostas de mudança são registrados como *issues*.

## 8. Instalação

- Requer Python 3.11 ou mais recente.
- Instalável por `pip install git+https://github.com/ml1-impatech/statstech.git`.

