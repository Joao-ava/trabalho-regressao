import numpy as np
from data import load_quarterback, load_radiation
from linear_regression import SingleLinearRegression, MultiLinearRegression
from metrics import r2, r2_adj, rmse, mae
from plots import RegressionPlotter
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
    print(f"{i+1:>4} {y_qb[i]:>12.4f} {y_pred_qb[i]:>12.4f} {y_qb[i] - y_pred_qb[i]:>14.4f}")
 
r2_qb = r2(y_qb, y_pred_qb)
rmse_qb = rmse(y_qb, y_pred_qb)
mae_qb = mae(y_qb, y_pred_qb)
print(f"\nR²   = {r2_qb:.4f}")
print(f"RMSE = {rmse_qb:.4f}")
print(f"MAE  = {mae_qb:.4f}")

# Grafico Scatter
plots = RegressionPlotter(x_qb, y_qb, model_simple,r2_qb)
plots.plot_single()

print()
# Problema 2: Regressão Linear Múltipla
print("=-" * 30)
print("Problema 2: Regressão Linear Múltipla")
print("=-" * 30)

# x_rad: colunas [mAmp, Tempo_de_Exposicao]
# y_rad: Dose_de_Radiacao
x_rad, y_rad = load_radiation()
 
# (A) Modelo completo: Dose ~ mAmp + Tempo
model_full = MultiLinearRegression()
y_pred_full = model_full.fit(x_rad, y_rad)
b = model_full.b

print("\n(A) Modelo de Regressão Linear Múltipla")
print(f"Intercepto (β₀): {b[0]:.4f}")
print(f"Coef. mAmp (β₁): {b[1]:.4f}")
print(f"Coef. Tempo  (β₂): {b[2]:.4f}")
print(f"Equação: Dose = {b[0]:.4f} + {b[1]:.4f} * mAmp + {b[2]:.4f} * Tempo")

# (B) Prever a dose de radiação quando a corrente for 15 mAmp e o tempo de exposição igual a 5 minutos
x_pred = np.array([[15.0, 5.0]])
y_pred_b = model_full.predict(x_pred)
print(f"\n(B) Previsão para corrente = 15 mAmp e tempo = 5 min")
print(f"Dose estimada: {y_pred_b[0]:.4f} rad")
 
# (C) Valor de R² score
r2_full = r2(y_rad, y_pred_full)
print(f"\n(C) R² Score do modelo completo")
print(f"R² = {r2_full:.6f}")
 
# (D) R² ajustado
n = len(y_rad)
p_full = 2  # dois preditores: mAmp e Tempo
r2_adj_full = r2_adj(y_rad, y_pred_full, p_full)
print(f"\n(D) R² Ajustado")
print(f"R² = {r2_full:.6f}")
print(f"R² adj = {r2_adj_full:.6f}")
print(f"O R² ajustado penaliza a adição de variáveis que não contribuem")
print(f"significativamente. Enquanto o R² comum sempre aumenta (ou permanece")
print(f"igual) ao adicionar preditores, o R² ajustado só aumenta se o preditor")
print(f"realmente melhora o modelo. Neste caso, ambos estão muito próximos,")
print(f"sugerindo que os dois preditores são relevantes.")

# Gráfico Scatter
plots = RegressionPlotter(x_rad, y_rad, model_full,r2_full)
plots.plot_multi()

# (E) Modelo alternativo: apenas mAmp
x_amp = x_rad[:, 0:1]  # só a coluna mAmp
model_amp = MultiLinearRegression()
y_pred_amp = model_amp.fit(x_amp, y_rad)
b_amp = model_amp.b
 
r2_amp = r2(y_rad, y_pred_amp)
r2_adj_amp = r2_adj(y_rad, y_pred_amp, p=1)
rmse_amp = rmse(y_rad, y_pred_amp)
mae_amp = mae(y_rad, y_pred_amp)
 
rmse_full = rmse(y_rad, y_pred_full)
mae_full = mae(y_rad, y_pred_full)
 
print(f"\n(E) Comparação: Modelo Completo vs Modelo apenas com mAmp")
print(f"{'Métrica':<12} {'Modelo Completo':>18} {'Só mAmp':>12}")
print(f"{'-'*44}")
print(f"{'R²':<12} {r2_full:>18.6f} {r2_amp:>12.6f}")
print(f"{'R² adj':<12} {r2_adj_full:>18.6f} {r2_adj_amp:>12.6f}")
print(f"{'RMSE':<12} {rmse_full:>18.4f} {rmse_amp:>12.4f}")
print(f"{'MAE':<12} {mae_full:>18.4f} {mae_amp:>12.4f}")
print()
print(f"Coef. β₀: {b_amp[0]:.4f}, β₁(mAmp): {b_amp[1]:.4f}")
print(f"O modelo completo (mAmp + Tempo) é melhor: possui R² e R² ajustado")
print(f"maiores e erros (RMSE, MAE) menores. Isso indica que o Tempo de")
print(f"Exposição contribui significativamente para explicar a Dose.")

# (f) Modelo com intercepto forçado a zero
# Forçar intercepto = 0: não adicionar coluna de 1s, resolver com np.linalg.lstsq
print(f"\n(f) Modelo com Intercepto Forçado a Zero")

# Sem intercepto: ŷ = β₁ * mAmp + β₂ * Tempo
b_zero = np.linalg.lstsq(x_rad, y_rad, rcond=None)[0]
y_pred_zero = x_rad @ b_zero

r2_zero = r2(y_rad, y_pred_zero)
rmse_zero = rmse(y_rad, y_pred_zero)
mae_zero = mae(y_rad, y_pred_zero)

print(f"Coef. mAmp (β₁) : {b_zero[0]:.4f}")
print(f"Coef. Tempo (β₂): {b_zero[1]:.4f}")
print(f"Equação: Dose = {b_zero[0]:.4f} * mAmp + {b_zero[1]:.4f} * Tempo")
print(f"\n{'Métrica':<12} {'Com Intercepto':>16} {'Sem Intercepto':>16}")
print(f"{'-' * 46}")
print(f"{'R²':<12} {r2_full:>16.6f} {r2_zero:>16.6f}")
print(f"{'RMSE':<12} {rmse_full:>16.4f} {rmse_zero:>16.4f}")
print(f"{'MAE':<12} {mae_full:>16.4f} {mae_zero:>16.4f}")
print()
print(f"Interpretação: forçar intercepto = 0 significa assumir que, quando")
print(f"corrente=0 e tempo=0, a dose de radiação é exatamente zero, o que")
print(f"faz sentido físico neste caso (sem exposição, sem dose).")
if r2_full >= r2_zero:
    print(f"Porém, o modelo com intercepto apresenta R² maior e RMSE/MAE menores,")
    print(f"sendo mais preciso. É preferível o modelo COM intercepto.")
else:
    print(f"O modelo sem intercepto apresenta desempenho comparável ou melhor,")
    print(f"e a restrição teórica é justificável neste domínio.")

# (H) Métricas adicionais: MSE, RMSE e MAE para modelo completo e só mAmp
mse_full = rmse_full ** 2
mse_amp = rmse_amp ** 2

print(f"\n(H) Métricas de Erro: Modelo Completo vs Apenas mAmp")
print(f"{'Métrica':<12} {'Modelo Completo':>18} {'Só mAmp':>12}")
print(f"{'-'*44}")
print(f"{'MSE':<12} {mse_full:>18.4f} {mse_amp:>12.4f}")
print(f"{'RMSE':<12} {rmse_full:>18.4f} {rmse_amp:>12.4f}")
print(f"{'MAE':<12} {mae_full:>18.4f} {mae_amp:>12.4f}")
print()
print(f"MSE (Erro Quadrático Médio): penaliza erros grandes com mais força.")
print(f"RMSE (Raiz do MSE): na mesma unidade da variável resposta (rad).")
print(f"MAE (Erro Absoluto Médio): mais robusto a outliers que o RMSE.")
print(f"Em todas as métricas o modelo completo supera o modelo só com mAmp,")
print(f"confirmando que incluir o Tempo de Exposição melhora as previsões.")