import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import log



years = list(range(1961, 2024))
vals = [
    36.47, 36.76, 37.06, 37.37, 37.77, 38.15, 38.54, 38.92, 39.41, 39.90,
    40.40, 40.89, 41.49, 42.19, 42.68, 43.31, 43.94, 44.57, 45.06, 45.55,
    46.11, 47.14, 48.24, 49.48, 50.74, 51.39, 51.95, 52.55, 53.20, 53.86,
    54.47, 54.78, 55.20, 55.69, 55.69, 55.69, 55.69, 55.69, 55.69, 55.69,
    55.76, 55.82, 55.88, 55.95, 56.01, 56.07, 56.14, 56.20, 56.26, 56.18,
    56.10, 56.02, 55.94, 55.86, 55.78, 55.70, 55.62, 55.54, 55.46, 55.45,
    55.44, 55.43, 55.43
]

arr = np.array(vals)
t = np.arange(1, 64)
n = len(vals)

plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

# =============================================================================
# 1.3 Основные характеристики
# =============================================================================
print("=" * 60)
print("1.3 ОСНОВНЫЕ ХАРАКТЕРИСТИКИ ВРЕМЕННОГО РЯДА")
print("=" * 60)
print(f"Число уровней: {n}")
print(f"Среднее значение: {arr.mean():.2f} %")
print(f"Медиана: {np.median(arr):.2f} %")
print(f"Стандартное отклонение: {arr.std():.2f} %")
print(f"Дисперсия: {arr.var():.2f}")
print(f"Минимум: {arr.min():.2f} % ({years[arr.argmin()]})")
print(f"Максимум: {arr.max():.2f} % ({years[arr.argmax()]})")
print(f"Размах: {arr.max() - arr.min():.2f} %")

# =============================================================================
# 1.4 Рисунок 1 — Линейный график
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, vals, color='steelblue', linewidth=1.5, marker='o', markersize=2)
ax.set_xlabel('Год')
ax.set_ylabel('Доля с/х земель, %')
ax.set_title('Временной ряд доли сельскохозяйственных земель Китая (1961–2023)')
ax.grid(True, alpha=0.3)
ax.set_xlim(1960, 2024)
plt.tight_layout()
plt.savefig('fig1.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nРисунок 1 сохранён: fig1.png")

# =============================================================================
# 1.4 Рисунок 2 — Столбчатая диаграмма (каждые 5 лет)
# =============================================================================
yrs5 = list(range(1961, 2024, 5))
vals5 = [vals[years.index(y)] for y in yrs5]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(yrs5, vals5, width=3.5, color='steelblue', edgecolor='white')
ax.set_xlabel('Год')
ax.set_ylabel('Доля с/х земель, %')
ax.set_title('Доля сельскохозяйственных земель Китая (каждые 5 лет)')
ax.set_ylim(0, 60)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 2 сохранён: fig2.png")

# =============================================================================
# 1.5 Автокорреляции и коррелограмма
# =============================================================================
def acf(x, maxlag=20):
    xm = x.mean()
    den = np.sum((x - xm) ** 2)
    return [np.sum((x[k:] - xm) * (x[:-k] - xm)) / den if k > 0 else 1.0
            for k in range(maxlag + 1)]

acf_vals = acf(arr, 20)
bound = 1.96 / np.sqrt(n)

print("\n" + "=" * 60)
print("1.5 АВТОКОРРЕЛЯЦИИ")
print("=" * 60)
print(f"Граница значимости: ±{bound:.3f}")
for k in range(1, 13):
    flag = "*" if abs(acf_vals[k]) > bound else ""
    print(f"  r{k:2d} = {acf_vals[k]:.3f} {flag}")

# Рисунок 3 — Коррелограмма
lags = list(range(21))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(lags, acf_vals, color='steelblue', width=0.6)
ax.axhline(bound, color='red', linestyle='--', linewidth=1,
           label=f'Граница значимости ±{bound:.3f}')
ax.axhline(-bound, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Лаг')
ax.set_ylabel('Автокорреляция')
ax.set_title('Коррелограмма временного ряда (ACF)')
ax.legend()
ax.set_ylim(-0.3, 1.1)
plt.tight_layout()
plt.savefig('fig3.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nРисунок 3 сохранён: fig3.png")

# ADF test (упрощённая версия через разности)
diff = np.diff(arr)
acf_diff = acf(diff, 5)
print(f"\nADF: ACF первых разностей r1={acf_diff[1]:.3f} (для формального теста используйте statsmodels)")

# =============================================================================
# 1.6 Сезонная и циклическая составляющие
# =============================================================================
print("\n" + "=" * 60)
print("1.6 СЕЗОННАЯ И ЦИКЛИЧЕСКАЯ СОСТАВЛЯЮЩИЕ")
print("=" * 60)
print("Сезонная составляющая: отсутствует (годовые данные)")

# Скользящая средняя для выделения тренда
def moving_average(x, m):
    k = m // 2
    return [(i, np.mean(x[i - k:i + k + 1])) for i in range(k, len(x) - k)]

ma5 = moving_average(vals, 5)
cyclic = [vals[i] - v for i, v in ma5]
print(f"Циклическая компонента: мин={min(cyclic):.3f}, макс={max(cyclic):.3f}")
print("Устойчивой периодичности не выявлено.")

# =============================================================================
# 1.7 Аномальные уровни (метод 3σ)
# =============================================================================
mean_ = arr.mean()
std_ = arr.std()
low3 = mean_ - 3 * std_
high3 = mean_ + 3 * std_

print("\n" + "=" * 60)
print("1.7 АНОМАЛЬНЫЕ УРОВНИ (метод 3σ)")
print("=" * 60)
print(f"ȳ = {mean_:.2f} %, σ = {std_:.2f} %")
print(f"Границы: [{low3:.2f} %; {high3:.2f} %]")
anomalies = [(years[i], v) for i, v in enumerate(vals) if v < low3 or v > high3]
if anomalies:
    print(f"Аномалии: {anomalies}")
else:
    print("Аномальные уровни отсутствуют.")

# =============================================================================
# 1.8 Сглаживание скользящей средней
# =============================================================================
ma7 = moving_average(vals, 7)

print("\n" + "=" * 60)
print("1.8 СГЛАЖИВАНИЕ СКОЛЬЗЯЩЕЙ СРЕДНЕЙ")
print("=" * 60)
print("Фрагмент MA(5):")
for i, v in ma5[:10]:
    print(f"  {years[i]}: {vals[i]:.2f} -> {v:.4f}")
print("  ...")
for i, v in ma5[-5:]:
    print(f"  {years[i]}: {vals[i]:.2f} -> {v:.4f}")

# Рисунок 4 — Сглаживание
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, vals, color='steelblue', linewidth=1, label='Исходный ряд', alpha=0.7)
ax.plot([years[i] for i, v in ma5], [v for i, v in ma5],
        color='red', linewidth=2, label='Скользящая средняя (m=5)')
ax.plot([years[i] for i, v in ma7], [v for i, v in ma7],
        color='green', linewidth=2, linestyle='--', label='Скользящая средняя (m=7)')
ax.set_xlabel('Год')
ax.set_ylabel('Доля с/х земель, %')
ax.set_title('Сглаживание временного ряда скользящей средней (m=5 и m=7)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nРисунок 4 сохранён: fig4.png")

# =============================================================================
# 1.9 Модели тренда
# =============================================================================
print("\n" + "=" * 60)
print("1.9 МОДЕЛИ ТРЕНДА")
print("=" * 60)

def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def durbin_watson(residuals):
    return np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)

# Линейная
c_lin = np.polyfit(t, arr, 1)
y_lin = np.polyval(c_lin, t)
print(f"Линейная: Ŷ(t) = {c_lin[1]:.4f} + {c_lin[0]:.4f}·t")
print(f"  R² = {r2(arr, y_lin):.4f}")

# Параболическая
c_par = np.polyfit(t, arr, 2)
y_par = np.polyval(c_par, t)
resid_par = arr - y_par
dw_par = durbin_watson(resid_par)
print(f"\nПараболическая: Ŷ(t) = {c_par[2]:.4f} + {c_par[1]:.4f}·t + ({c_par[0]:.6f})·t²")
print(f"  R² = {r2(arr, y_par):.4f}")
print(f"  MAE = {mae(arr, y_par):.3f} %")
print(f"  MAPE = {mape(arr, y_par):.3f} %")
print(f"  DW = {dw_par:.3f}")

# Логарифмическая
c_log = np.polyfit(log(t), arr, 1)
y_log = c_log[0] * log(t) + c_log[1]
print(f"\nЛогарифмическая: Ŷ(t) = {c_log[1]:.4f} + {c_log[0]:.4f}·ln(t)")
print(f"  R² = {r2(arr, y_log):.4f}")

# Рисунок 5 — Линии тренда
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, vals, color='steelblue', linewidth=1.5, label='Исходный ряд')
ax.plot(years, y_lin, color='red', linewidth=1.5,
        label=f'Линейный (R²={r2(arr, y_lin):.4f})')
ax.plot(years, y_par, color='green', linewidth=1.5, linestyle='--',
        label=f'Параболический (R²={r2(arr, y_par):.4f})')
ax.plot(years, y_log, color='orange', linewidth=1.5, linestyle=':',
        label=f'Логарифмический (R²={r2(arr, y_log):.4f})')
ax.set_xlabel('Год')
ax.set_ylabel('Доля с/х земель, %')
ax.set_title('Линии тренда временного ряда (линейная, параболическая, логарифмическая)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nРисунок 5 сохранён: fig5.png")

# Рисунок 6 — Остатки
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(years, resid_par, color='steelblue', width=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Год')
ax.set_ylabel('Остаток')
ax.set_title('Остаточная компонента (отклонения от параболического тренда)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig6.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 6 сохранён: fig6.png")

# =============================================================================
# 2.1 Коррелограмма остатков
# =============================================================================
acf_resid = acf(resid_par, 15)
print("\n" + "=" * 60)
print("2.1 ОСТАТОЧНАЯ КОМПОНЕНТА")
print("=" * 60)
print(f"r1 остатков = {acf_resid[1]:.3f}, r2 = {acf_resid[2]:.3f}")

lags2 = list(range(16))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(lags2, acf_resid, color='steelblue', width=0.6)
ax.axhline(bound, color='red', linestyle='--', linewidth=1,
           label=f'Граница значимости ±{bound:.3f}')
ax.axhline(-bound, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Лаг')
ax.set_ylabel('Автокорреляция')
ax.set_title('Коррелограмма остаточной компоненты')
ax.legend()
plt.tight_layout()
plt.savefig('fig7.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 7 сохранён: fig7.png")

# =============================================================================
# 2.2 Сравнение методов сглаживания
# =============================================================================
print("\n" + "=" * 60)
print("2.2 СРАВНЕНИЕ МЕТОДОВ СГЛАЖИВАНИЯ")
print("=" * 60)

ma5_arr = np.array([v for i, v in ma5])
ma5_idx = [i for i, v in ma5]
ma7_arr = np.array([v for i, v in ma7])
ma7_idx = [i for i, v in ma7]

rmse5 = np.sqrt(np.mean((arr[ma5_idx] - ma5_arr) ** 2))
rmse7 = np.sqrt(np.mean((arr[ma7_idx] - ma7_arr) ** 2))
print(f"MA(5): n={len(ma5)}, RMSE={rmse5:.4f} %")
print(f"MA(7): n={len(ma7)}, RMSE={rmse7:.4f} %")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(years, vals, color='steelblue', linewidth=1, label='Исходный ряд', alpha=0.6)
ax.plot([years[i] for i, v in ma5], [v for i, v in ma5],
        color='red', linewidth=2, label='Скользящая средняя (m=5)')
ax.plot([years[i] for i, v in ma7], [v for i, v in ma7],
        color='green', linewidth=2, linestyle='--', label='Скользящая средняя (m=7)')
ax.set_xlabel('Год')
ax.set_ylabel('Доля с/х земель, %')
ax.set_title('Сравнение методов сглаживания (m=5 и m=7)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig8.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 8 сохранён: fig8.png")

print("\n" + "=" * 60)
print("Все расчёты выполнены. Графики сохранены в текущую папку.")
print("=" * 60)
