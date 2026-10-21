import numpy as np

from statstech.model_selection.bootstrap import bootstrap

rng = np.random.default_rng(0)

# Dados: 50 observações de uma distribuição com média 10 e desvio 2.
data = rng.normal(loc=10.0, scale=2.0, size=50)


# Estatística simples: a média
estimates, std_error = bootstrap(data, np.mean, random_state=0)
low, high = np.percentile(estimates, [2.5, 97.5])

print("Média de amostra:", round(data.mean(), 3))
print("Estimativas bootstrap:", estimates.shape)
print("Erro padrão (bootstrap):", round(std_error, 3))
print("Erro padrão (fórmula s / raiz de n):", round(data.std(ddof=1) / 50**0.5, 3))
print("Intervalo de 95%:", np.round([low, high], 3))

# Estatística sem fórmula simples: a mediana
estimates, std_error = bootstrap(data, np.median, n_bootstrap=1000, random_state=0)

print("\nMediana da amostra:", round(np.median(data), 3))
print("Erro padrão da mediana:", round(std_error, 3))

# Estatística de dois arrays: passe os dados como uma tupla.
x = rng.normal(size=50)
y = 2 * x + rng.normal(scale=0.5, size=50)


def correlação(a, b):
    return np.corrcoef(a, b)[0, 1]


estimates, std_error = bootstrap((x, y), correlação, n_bootstrap=1000, random_state=0)
low, high = np.percentile(estimates, [2.5, 97.5])

print("\nCorrelação da amostra:", round(correlação(x, y), 3))
print("Erro padrão da correlação:", round(std_error, 3))
print("Intervalo de 95%:", np.round([low, high], 3))

# Estatística que devolve vários valores: o erro padrão vem ver um por valor.
X = rng.normal(loc=[0.0, 5.0], scale=[1.0, 3.0], size=(50, 2))
estimates, std_error = bootstrap(
    X, lambda a: a.mean(axis=0), n_bootstrap=1000, random_state=0
)

print("\nMédia de cada coluna:", np.round(X.mean(axis=0), 3))
print("Formato das estimativas:", estimates.shape)
print("Erro padrão de cada coluna:", np.round(std_error, 3))
