import numpy as np
from data import load_quarterback, load_radiation
from linear_regression import SingleLinearRegression, MultiLinearRegression
from metrics import r2, r2_adj, rmse, mae

# Problema 1: Regressão Linear Simples 

print("=-" * 30)
print("Problema 1: Regressão Linear Simples")
print("=-" * 30)

x_qb, y_qb = load_quarterback()
model_simple = SingleLinearRegression()
y_pred_qb = model_simple.fit(x_qb, y_qb)

# (A) Estimativa de mínimos quadrados: inclinação b1 e interseção b0
print("\n(A) Estimativas de mínimos quadrados:")
print(f"Inclinação (b1): {model_simple.b1:.4f}")
print(f"Interseção (b0): {model_simple.b0:.4f}")
print(f"Equação do modelo: ŷ = {model_simple.b0:.4f} + {model_simple.b1:.4f} * x")

# (B) Estimativa da Pontuação Média para x = 7.5 jardas por tentativa
x_b = np.array([7.5])
y_b = model_simple.predict(x_b)
print(f"\n(B) Estimativa para x = 7.5 jardas por tentativa")
print(f"Pontuação Estimada: ŷ = {y_b[0]:.4f}")

# (C) Mudança na pontuação associada a uma diminuição de 1 jarda por tentativa
print(f"\n(C) Mudança na pontuação para diminuição de 1 jarda/tentativa")
print(f"A inclinação β₁ = {model_simple.b1:.4f} indica que, para cada")
print(f"aumento de 1 jarda/tentativa, a pontuação média aumenta {model_simple.b1:.4f}.")
print(f"Logo, uma diminuição de 1 jarda implica uma mudança de {-model_simple.b1:.4f} na pontuação.")

# (D) Para x = 7,21, encontrar o valor ajustado e o resíduo correspondente
print(f"\n(D) Valor ajustado e resíduo para x = 7,21 jardas")
mask = x_qb == 7.21
x_d = x_qb[mask]
y_d = y_qb[mask]
y_pred_d = model_simple.predict(x_d)
residuos_d = y_d - y_pred_d
for i in range(len(x_d)):
    print(f"Observação com x = 7.21:")
    print(f"=> y observado  : {y_d[i]:.4f}")
    print(f"=> ŷ ajustado   : {y_pred_d[i]:.4f}")
    print(f"=> Resíduo (e_i) : {residuos_d[i]:.4f}")
    
# (E) Análise de resíduos: tabela completa
print(f"\n(E) Tabela de Resíduos")
print(f"{'i':>4} {'y_i (obs)':>12} {'ŷ_i (ajust)':>12} {'e_i (resíduo)':>14}")
print("" + "-" * 46)
for i in range(len(y_qb)):
    print(f"    {i+1:>4} {y_qb[i]:>12.4f} {y_pred_qb[i]:>12.4f} {y_qb[i] - y_pred_qb[i]:>14.4f}")
 
r2_qb = r2(y_qb, y_pred_qb)
rmse_qb = rmse(y_qb, y_pred_qb)
mae_qb = mae(y_qb, y_pred_qb)
print(f"\nR²   = {r2_qb:.4f}")
print(f"RMSE = {rmse_qb:.4f}")
print(f"MAE  = {mae_qb:.4f}")