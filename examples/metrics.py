"""Exemplos de uso das métricas de regressão, classificação e scorers.

Execute a partir da raiz do projeto:
    python examples.py

O exemplo pressupõe que o registro de scorers esteja em
statstech.metrics.scoring. Se o arquivo tiver outro nome, ajuste esse import.
"""

import numpy as np

import statstech.metrics.classification as cl
import statstech.metrics.regression as rg
import statstech.metrics.scorers as scoring


def example_regression() -> None:
    """Demonstra métricas que recebem diretamente y_true e y_pred."""
    print("\n" + "=" * 60)
    print("1. MÉTRICAS DE REGRESSÃO")
    print("=" * 60)

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    print(f"Valores reais:   {y_true}")
    print(f"Valores previstos: {y_pred}")
    print(f"RSS:  {rg.residual_sum_squares(y_true, y_pred):.4f}")
    print(f"MSE:  {rg.mean_squared_error(y_true, y_pred):.4f}")
    print(f"MAE:  {rg.mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {rg.root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"R²:   {rg.r2_score(y_true, y_pred):.4f}")


def example_classification() -> tuple[list[int], list[int], list[int], np.ndarray]:
    """Demonstra a matriz de confusão e métricas derivadas dela."""
    print("\n" + "=" * 60)
    print("2. MÉTRICAS DE CLASSIFICAÇÃO")
    print("=" * 60)

    classes = [0, 1, 2]
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 1, 2, 1, 1, 2]

    # As linhas representam as classes reais e as colunas, as previstas.
    c_matrix = cl.confusion_matrix(y_true, y_pred, classes)

    print("\nMatriz de confusão (linhas = reais; colunas = previstas):")
    for row in c_matrix:
        print(row)

    # Estas funções recebem c_matrix, e não y_true/y_pred diretamente.
    class_k = 1

    print(f"\nMétricas para a classe {class_k}:")
    print(f"TP: {cl.true_positives(c_matrix, class_k)}")
    print(f"FP: {cl.false_positives(c_matrix, class_k)}")
    print(f"TN: {cl.true_negatives(c_matrix, class_k)}")
    print(f"FN: {cl.false_negatives(c_matrix, class_k)}")
    print(f"Precisão:       {cl.precision_score(c_matrix, class_k):.4f}")
    print(f"Revocação:      {cl.recall_score(c_matrix, class_k):.4f}")
    print(f"Especificidade: {cl.specificity_score(c_matrix, class_k):.4f}")
    print(f"F1:             {cl.f1_score(c_matrix, class_k):.4f}")

    print("\nMétricas gerais/macro (também recebem c_matrix):")
    print(f"Acurácia:             {cl.accuracy_score(c_matrix):.4f}")
    print(f"Precisão macro:       {cl.macro_precision(c_matrix):.4f}")
    print(f"Revocação macro:      {cl.macro_recall(c_matrix):.4f}")
    print(f"Especificidade macro: {cl.macro_specificity(c_matrix):.4f}")
    print(f"F1 macro:             {cl.macro_f1(c_matrix):.4f}")

    # Para log loss, a entrada é uma matriz de probabilidades, não c_matrix.
    # Cada linha deve somar 1, e cada coluna corresponde a uma classe.
    p_matrix = np.array(
        [
            [0.80, 0.10, 0.10],  # verdadeiro 0; previsão 0
            [0.10, 0.20, 0.70],  # verdadeiro 1; previsão 2
            [0.10, 0.70, 0.20],  # verdadeiro 2; previsão 1
            [0.75, 0.15, 0.10],  # verdadeiro 0; previsão 0
            [0.10, 0.80, 0.10],  # verdadeiro 1; previsão 1
            [0.05, 0.15, 0.80],  # verdadeiro 2; previsão 2
            [0.35, 0.50, 0.15],  # verdadeiro 0; previsão 1
            [0.10, 0.70, 0.20],  # verdadeiro 1; previsão 1
            [0.10, 0.20, 0.70],  # verdadeiro 2; previsão 2
        ]
    )

    print("\nLog loss (usa probabilidades por observação):")
    print(f"Log loss: {cl.log_loss(y_true, p_matrix, classes=classes):.4f}")

    # ROC-AUC binária recebe scores da classe positiva.
    y_binary = [0, 0, 1, 1]
    positive_scores = [0.10, 0.40, 0.35, 0.80]

    print("\nROC-AUC binária:")
    print(f"ROC-AUC: {cl.roc_auc_score(y_binary, positive_scores):.4f}")

    # Em multiclasse, a função recebe uma matriz de scores e class_k.
    # A classe escolhida é comparada com todas as outras (One-vs-Rest).
    print("\nROC-AUC multiclasse (classe 1 contra as demais):")
    print(f"ROC-AUC: {cl.roc_auc_score(y_true, p_matrix, class_k=1):.4f}")

    return classes, y_true, y_pred, p_matrix


def example_scorer_registry(
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    classes: list[int],
    y_cls_true: list[int],
    y_cls_pred: list[int],
    p_matrix: np.ndarray,
) -> None:
    """Demonstra como obter scorers pelo nome no registro."""
    print("\n" + "=" * 60)
    print("3. REGISTRO DE SCORERS")
    print("=" * 60)

    # O registro permite selecionar uma métrica por uma string.
    rmse_scorer = scoring.get_scorer("rmse")
    accuracy_scorer = scoring.get_scorer("accuracy")
    log_loss_scorer = scoring.get_scorer("log_loss")

    print(f"Scorer 'rmse':     {rmse_scorer(y_reg_true, y_reg_pred):.4f}")
    print(f"Scorer 'accuracy': {accuracy_scorer(y_cls_true, y_cls_pred):.4f}")
    print(
        "Scorer 'log_loss': "
        f"{log_loss_scorer(y_cls_true, p_matrix, classes=classes):.4f}"
    )

    print("\nMetadados do registro:")
    print(f"'rmse' tem menor valor como objetivo: {scoring.lower_is_better('rmse')}")
    print(
        f"'accuracy' tem menor valor como objetivo: "
        f"{scoring.lower_is_better('accuracy')}"
    )
    print(f"'log_loss' precisa de probabilidades: {scoring.needs_proba('log_loss')}")
    print(f"'accuracy' precisa de probabilidades: {scoring.needs_proba('accuracy')}")

    # O adaptador do registro transforma (y_true, y_pred) em c_matrix
    # antes de chamar accuracy_score. Por isso, não passamos c_matrix aqui.
    # Já log_loss exige probabilidades; seu scorer recebe p_matrix.


def main() -> None:
    example_regression()

    # Reutilizamos os mesmos dados para mostrar como o registro de scorers
    # oferece uma interface comum para métricas com entradas diferentes.
    classes, y_cls_true, y_cls_pred, p_matrix = example_classification()

    y_reg_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_reg_pred = np.array([2.5, 0.0, 2.0, 8.0])

    example_scorer_registry(
        y_reg_true,
        y_reg_pred,
        classes,
        y_cls_true,
        y_cls_pred,
        p_matrix,
    )


if __name__ == "__main__":
    main()
