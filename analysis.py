

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# ДАННЫЕ
# =============================================================================

years = list(range(1961, 2024))

values = [
    36.465795, 36.758714, 37.062285, 37.365856, 37.770618,
    38.175380, 38.580143, 38.984905, 39.389667, 39.794430,
    40.199192, 40.603954, 41.008717, 41.413479, 41.818241,
    42.223004, 42.627766, 43.032528, 43.437291, 43.842053,
    44.246815, 44.651578, 45.056340, 45.461102, 45.865865,
    46.270627, 46.675389, 47.080152, 47.484914, 47.889676,
    48.294439, 48.699201, 49.103963, 49.508726, 49.913488,
    50.318250, 50.723013, 51.127775, 51.532537, 51.937300,
    52.342062, 52.746824, 53.151587, 53.556349, 53.961111,
    54.365874, 54.770636, 55.175398, 55.580161, 55.580161,
    55.570149, 55.560136, 55.550124, 55.540112, 55.530099,
    55.520087, 55.510075, 55.500062, 55.490050, 55.480037,
    55.470025, 55.452637, 55.442624
]

y = np.array(values)
x = np.array(years)
n = len(y)
t = np.arange(1, n + 1)

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})

# =============================================================================
# РИСУНОК 1 — Временной ряд
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, color='steelblue', linewidth=1.8, marker='o', markersize=3)
ax.set_xlabel('Год')
ax.set_ylabel('Доля сельскохозяйственных земель, %')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(1961, 2023)
plt.tight_layout()
plt.savefig('fig1_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 1 сохранён")

# =============================================================================
# РИСУНОК 2 — Столбчатая диаграмма (каждые 5 лет)
# =============================================================================

idx5 = list(range(0, n, 5))
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar([years[i] for i in idx5], [y[i] for i in idx5], color='steelblue', width=3.5)
ax.set_xlabel('Год')
ax.set_ylabel('Доля сельскохозяйственных земель, %')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig2_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 2 сохранён")

# =============================================================================
# РИСУНОК 3 — Коррелограмма (ACF)
# =============================================================================

max_lag = 20
acf_vals = []
y_mean = y.mean()
c0 = np.sum((y - y_mean) ** 2) / n
for lag in range(0, max_lag + 1):
    ck = np.sum((y[:n - lag] - y_mean) * (y[lag:] - y_mean)) / n
    acf_vals.append(ck / c0)

conf = 1.96 / np.sqrt(n)
print(f"\nГраница значимости ACF: ±{conf:.4f}")
print(f"ACF лаги 1–5: {[round(v, 4) for v in acf_vals[1:6]]}")

fig, ax = plt.subplots(figsize=(11, 5))
lags = range(0, max_lag + 1)
ax.bar(lags, acf_vals, color='steelblue', width=0.6)
ax.axhline(y=conf, color='red', linestyle='--', linewidth=1.2,
           label=f'Граница значимости ±{conf:.3f}')
ax.axhline(y=-conf, color='red', linestyle='--', linewidth=1.2)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Лаг')
ax.set_ylabel('Автокорреляция')
ax.legend()
ax.set_xticks(list(lags))
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('fig3_acf.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 3 сохранён")

# =============================================================================
# ПРОВЕРКА СТАЦИОНАРНОСТИ — тест Дики–Фуллера (упрощённый)
# =============================================================================

dy = np.diff(y)
y_lag = y[:-1]
X_df = np.column_stack([np.ones(len(y_lag)), y_lag])
b_df = np.linalg.lstsq(X_df, dy, rcond=None)[0]
resid_df = dy - X_df @ b_df
s2_df = np.sum(resid_df ** 2) / (len(dy) - 2)
cov_df = s2_df * np.linalg.inv(X_df.T @ X_df)
se_b = np.sqrt(cov_df[1, 1])
t_adf = b_df[1] / se_b
print(f"\nТест Дики–Фуллера: t = {t_adf:.4f}")
print(f"Критическое значение (5%): -2.89")
print(f"Вывод: ряд {'стационарный' if t_adf < -2.89 else 'нестационарный (тренд)'}")

# =============================================================================
# АНОМАЛЬНЫЕ УРОВНИ — метод 3σ
# =============================================================================

y_std = y.std()
upper = y_mean + 3 * y_std
lower = y_mean - 3 * y_std
anomalies = [(years[i], y[i]) for i in range(n) if y[i] > upper or y[i] < lower]
print(f"\nМетод 3σ: [{lower:.4f}; {upper:.4f}]")
print(f"Аномалии: {anomalies if anomalies else 'не обнаружены'}")

# =============================================================================
# РИСУНОК 4 — Сглаживание скользящей средней (m=5 и m=7)
# =============================================================================

w5 = 5
ma5 = np.convolve(y, np.ones(w5) / w5, mode='valid')
ma5_years = years[w5 // 2: n - w5 // 2]

w7 = 7
ma7 = np.convolve(y, np.ones(w7) / w7, mode='valid')
ma7_years = years[w7 // 2: n - w7 // 2]

# Сравнение качества сглаживания
rmse5 = np.sqrt(np.mean((y[w5 // 2: n - w5 // 2] - ma5) ** 2))
rmse7 = np.sqrt(np.mean((y[w7 // 2: n - w7 // 2] - ma7) ** 2))
print(f"\nСкользящая средняя m=5: RMSE = {rmse5:.4f}")
print(f"Скользящая средняя m=7: RMSE = {rmse7:.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, color='steelblue', linewidth=1.5,
        label='Исходный ряд', marker='o', markersize=2.5)
ax.plot(ma5_years, ma5, color='red', linewidth=2,
        label='Скользящая средняя (m=5)')
ax.plot(ma7_years, ma7, color='green', linewidth=2, linestyle='--',
        label='Скользящая средняя (m=7)')
ax.set_xlabel('Год')
ax.set_ylabel('Доля сельскохозяйственных земель, %')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig4_ma.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 4 сохранён")

# =============================================================================
# РИСУНОК 5 — Линии тренда (линейная, параболическая, логарифмическая)
# =============================================================================

# Линейный тренд
slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(t, y)
y_lin = intercept_lin + slope_lin * t
ss_tot = np.sum((y - y_mean) ** 2)
r2_lin = 1 - np.sum((y - y_lin) ** 2) / ss_tot

# Параболический тренд
coeffs2 = np.polyfit(t, y, 2)
y_par = np.polyval(coeffs2, t)
r2_par = 1 - np.sum((y - y_par) ** 2) / ss_tot

# Логарифмический тренд
coeffs_log = np.polyfit(np.log(t), y, 1)
y_log = coeffs_log[0] * np.log(t) + coeffs_log[1]
r2_log = 1 - np.sum((y - y_log) ** 2) / ss_tot

print(f"\nЛинейный тренд: Y = {intercept_lin:.4f} + {slope_lin:.4f}*t,  R² = {r2_lin:.6f}")
print(f"Параболический: Y = {coeffs2[2]:.4f} + {coeffs2[1]:.4f}*t + {coeffs2[0]:.6f}*t²,  R² = {r2_par:.6f}")
print(f"Логарифмический: Y = {coeffs_log[1]:.4f} + {coeffs_log[0]:.4f}*ln(t),  R² = {r2_log:.6f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, color='steelblue', linewidth=1.5,
        label='Исходный ряд', marker='o', markersize=2.5)
ax.plot(x, y_lin, color='red', linewidth=2,
        label=f'Линейный (R²={r2_lin:.4f})')
ax.plot(x, y_par, color='green', linewidth=2, linestyle='--',
        label=f'Параболический (R²={r2_par:.4f})')
ax.plot(x, y_log, color='orange', linewidth=2, linestyle=':',
        label=f'Логарифмический (R²={r2_log:.4f})')
ax.set_xlabel('Год')
ax.set_ylabel('Доля сельскохозяйственных земель, %')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig5_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 5 сохранён")

# =============================================================================
# ОЦЕНКА АДЕКВАТНОСТИ — параболическая модель
# =============================================================================

resid_par = y - y_par
mae = np.mean(np.abs(resid_par))
mape = np.mean(np.abs(resid_par / y)) * 100
dw = np.sum(np.diff(resid_par) ** 2) / np.sum(resid_par ** 2)

print(f"\nАдекватность параболического тренда:")
print(f"MAE  = {mae:.4f} %")
print(f"MAPE = {mape:.4f} %")
print(f"DW   = {dw:.4f}")
print("Вывод: MAPE < 5% — высокая точность аппроксимации.")
print("       DW << 2 — сильная автокорреляция остатков,")
print("       модель не адекватна в строгом статистическом смысле.")

# =============================================================================
# РИСУНОК 6 — Остаточная компонента
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x, resid_par, color='steelblue', alpha=0.7)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Год')
ax.set_ylabel('Остаток')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig6_residuals.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 6 сохранён")

# =============================================================================
# РИСУНОК 7 — Коррелограмма остатков
# =============================================================================

resid = resid_par
r_mean = resid.mean()
c0r = np.sum((resid - r_mean) ** 2) / n
acf_resid = []
for lag in range(0, 16):
    ck = np.sum((resid[:n - lag] - r_mean) * (resid[lag:] - r_mean)) / n
    acf_resid.append(ck / c0r)

print(f"\nACF остатков лаги 1–5: {[round(v, 4) for v in acf_resid[1:6]]}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(16), acf_resid, color='steelblue', width=0.6)
ax.axhline(conf, color='red', linestyle='--', linewidth=1.2,
           label=f'Граница значимости ±{conf:.3f}')
ax.axhline(-conf, color='red', linestyle='--', linewidth=1.2)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Лаг')
ax.set_ylabel('Автокорреляция')
ax.legend()
ax.set_xticks(range(16))
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('fig7_acf_resid.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 7 сохранён")

print("\nВсе расчёты и графики выполнены успешно.")

# =============================================================================
# РИСУНОК 8 — Сравнение методов сглаживания (итоговый)
# (дублирует рисунок 4 — используется в разделе 2.2 дополнительной части)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, color='steelblue', linewidth=1.5,
        label='Исходный ряд', marker='o', markersize=2.5)
ax.plot(ma5_years, ma5, color='red', linewidth=2,
        label=f'Скользящая средняя (m=5), RMSE={rmse5:.4f}')
ax.plot(ma7_years, ma7, color='green', linewidth=2, linestyle='--',
        label=f'Скользящая средняя (m=7), RMSE={rmse7:.4f}')
ax.set_xlabel('Год')
ax.set_ylabel('Доля сельскохозяйственных земель, %')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig8_smoothing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Рисунок 8 сохранён")

print("\n=== Итоговые результаты ===")
print(f"Период: 1961–2023, n = {n}")
print(f"Среднее: {y_mean:.4f}%,  Std: {y.std():.4f}%")
print(f"Мин: {y.min():.4f}%,  Макс: {y.max():.4f}%")
print(f"Лучшая модель тренда: параболическая (R²={r2_par:.4f}, MAPE={mape:.4f}%)")
