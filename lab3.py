# Лабораторна робота №3: МГУА та інформаційний критерій
# Варіант 23
# Студентка: Тугаріна Емілія, група 304-ТН

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

print("="*60)
print("ЛАБОРАТОРНА РОБОТА №3: МГУА ТА ІНФОРМАЦІЙНИЙ КРИТЕРІЙ")
print("Варіант: 23 | Студентка: Тугаріна Емілія")
print("="*60)

# === 1. ГЕНЕРАЦІЯ ДАНИХ (ТІ Ж, ЩО Й У ЛАБОРАТОРНІЙ №2) ===
np.random.seed(23)
x = np.linspace(-3, 3, 100)
y = np.sin(x) + 0.5 * x + 0.2 * np.random.randn(100)

X = x.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

print(f"\n1. Використано ті самі дані, що й у лабораторній роботі №2:")
print(f"   Формула: y = sin(x) + 0.5*x + шум (N(0, 0.2))")
print(f"   Розмір вибірки: {len(x)} точок")
print(f"   Навчальна вибірка: {len(X_train)} точок (70%)")
print(f"   Тестова вибірка: {len(X_test)} точок (30%)")

# === 2. ФУНКЦІЯ ІНФОРМАЦІЙНОГО КРИТЕРІЮ ===
def information_criterion(mse, k, n):
    """Обчислення інформаційного критерію за формулою: I = σ² / (1 - k²/n)"""
    if k**2 >= n:
        return float('inf')  # Уникаємо ділення на 0 або від'ємне значення
    return mse / (1 - (k**2 / n))

# === 3. РЕАЛІЗАЦІЯ МГУА ДЛЯ ПОЛІНОМІАЛЬНИХ МОДЕЛЕЙ ===
print("\n2. Виконання МГУА для поліномів різних ступенів:")

results = []
max_degree = 8
n_train = len(X_train)

for degree in range(1, max_degree + 1):
    # Створення поліноміальних ознак
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Навчання моделі
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Прогнозування
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Обчислення метрик
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Кількість параметрів (коефіцієнтів) у моделі
    k = X_train_poly.shape[1]  # Кількість стовпців = кількість параметрів
    
    # Обчислення інформаційного критерію
    ic_train = information_criterion(train_mse, k, n_train)
    ic_test = information_criterion(test_mse, k, n_train)
    
    # Збереження результатів
    results.append({
        'degree': degree,
        'k': k,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'ic_train': ic_train,
        'ic_test': ic_test,
        'model': model,
        'poly': poly
    })
    
    print(f"   Поліном {degree}-го ступеня: k={k}, Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}, Test IC={ic_test:.4f}")

# === 4. ВИБІР ОПТИМАЛЬНОЇ МОДЕЛІ ЗА МГУА ===
# Вибір моделі з мінімальним інформаційним критерієм на тестовій вибірці
best_idx = np.argmin([r['ic_test'] for r in results])
best_gmdh = results[best_idx]

print(f"\n3. ОПТИМАЛЬНА МОДЕЛЬ ЗА МГУА (мінімальний IC):")
print(f"   Ступінь полінома: {best_gmdh['degree']}")
print(f"   Кількість параметрів (k): {best_gmdh['k']}")
print(f"   Інформаційний критерій (Test IC): {best_gmdh['ic_test']:.4f}")
print(f"   Test MSE: {best_gmdh['test_mse']:.4f}")
print(f"   Test R²: {best_gmdh['test_r2']:.4f}")

# === 5. ПОРІВНЯННЯ З РЕЗУЛЬТАТАМИ ЛАБОРАТОРНОЇ №2 ===
print(f"\n4. ПОРІВНЯННЯ З РЕЗУЛЬТАТАМИ ПОЛІНОМІАЛЬНОЇ РЕГРЕСІЇ (ЛАБ. №2):")

# Результати з лабораторної №2
lab2_results = {
    'degrees': [1, 2, 3, 6, 12],
    'test_mse': [0.2309, 0.1981, 0.1619, 0.1874, 0.3421],
    'test_r2': [0.8075, 0.8343, 0.8641, 0.8429, 0.7130]
}

# Найкраща модель з лабораторної №2 (поліном 3-го ступеня)
best_lab2_mse = min(lab2_results['test_mse'])
best_lab2_idx = lab2_results['test_mse'].index(best_lab2_mse)
best_lab2_degree = lab2_results['degrees'][best_lab2_idx]

print(f"   Найкраща модель з лаб. №2: поліном {best_lab2_degree}-го ступеня")
print(f"   Test MSE (лаб. №2): {best_lab2_mse:.4f}")
print(f"   Test MSE (МГУА): {best_gmdh['test_mse']:.4f}")

if abs(best_gmdh['test_mse'] - best_lab2_mse) < 0.01:
    print(f"   Висновок: МГУА знайшов ту саму оптимальну модель!")
else:
    print(f"   Висновок: Моделі відрізняються (різниця: {abs(best_gmdh['test_mse'] - best_lab2_mse):.4f})")

# === 6. ПОБУДОВА ГРАФІКІВ ===
print(f"\n5. Побудова графіків...")

plt.figure(figsize=(14, 10))

# Графік 1: Вихідні дані та оптимальна модель МГУА
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=30, label='Навчальні дані')
plt.scatter(X_test, y_test, color='red', alpha=0.6, s=30, marker='x', label='Тестові дані')

# Побудова кривої оптимальної моделі
x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
x_plot_poly = best_gmdh['poly'].transform(x_plot)
y_plot_pred = best_gmdh['model'].predict(x_plot_poly)
plt.plot(x_plot, y_plot_pred, 'g-', linewidth=3, label=f'МГУА (ступінь {best_gmdh["degree"]})')

plt.title('Оптимальна модель МГУА', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік 2: Залежність інформаційного критерію від ступеня полінома
plt.subplot(2, 2, 2)
degrees = [r['degree'] for r in results]
ic_train_vals = [r['ic_train'] for r in results]
ic_test_vals = [r['ic_test'] for r in results]

plt.plot(degrees, ic_train_vals, 'bo-', linewidth=2, markersize=8, label='IC (Train)')
plt.plot(degrees, ic_test_vals, 'ro-', linewidth=2, markersize=8, label='IC (Test)')
plt.axvline(x=best_gmdh['degree'], color='green', linestyle='--', linewidth=2, 
           label=f'Оптимальна (ступінь {best_gmdh["degree"]})')

plt.title('Залежність інформаційного критерію (IC) від ступеня полінома', fontsize=14)
plt.xlabel('Ступінь полінома')
plt.ylabel('Інформаційний критерій (IC)')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік 3: Порівняння MSE МГУА та поліноміальної регресії
plt.subplot(2, 2, 3)
mse_gmdh = [r['test_mse'] for r in results]

plt.plot(degrees, mse_gmdh, 'bo-', linewidth=2, markersize=8, label='МГУА (Test MSE)')
plt.plot(lab2_results['degrees'], lab2_results['test_mse'], 'ro-', linewidth=2, markersize=8, label='Поліном. регресія (Test MSE)')

plt.title('Порівняння Test MSE: МГУА vs Поліноміальна регресія', fontsize=14)
plt.xlabel('Ступінь полінома')
plt.ylabel('Test MSE')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік 4: Порівняння оптимальних моделей
plt.subplot(2, 2, 4)
plt.scatter(X_test, y_test, color='red', alpha=0.6, s=30, label='Тестові дані')

# Оптимальна модель МГУА
plt.plot(x_plot, y_plot_pred, 'g-', linewidth=3, label=f'МГУА (ступінь {best_gmdh["degree"]})')

# Найкраща модель з лаб. №2 (поліном 3-го ступеня)
poly_best = PolynomialFeatures(degree=3)
X_train_poly_best = poly_best.fit_transform(X_train)
model_best = LinearRegression()
model_best.fit(X_train_poly_best, y_train)

x_plot_poly_best = poly_best.transform(x_plot)
y_plot_pred_best = model_best.predict(x_plot_poly_best)
plt.plot(x_plot, y_plot_pred_best, 'b--', linewidth=2, label='Поліном. регресія (ступінь 3)')

plt.title('Порівняння оптимальних моделей', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab3_plot.png', dpi=150, bbox_inches='tight')
print(f"   Графік збережено: lab3_plot.png")

# === 7. ТАБЛИЦЯ РЕЗУЛЬТАТІВ ДЛЯ ВІДОБРАЖЕННЯ ===
print(f"\n6. ТАБЛИЦЯ РЕЗУЛЬТАТІВ МГУА:")
print("="*85)
print(f"{'Ступінь':<8} {'k':<6} {'Train MSE':<12} {'Test MSE':<12} {'Test R²':<12} {'Test IC':<12}")
print("-"*85)

for r in results:
    print(f"{r['degree']:<8} {r['k']:<6} {r['train_mse']:<12.4f} {r['test_mse']:<12.4f} {r['test_r2']:<12.4f} {r['ic_test']:<12.4f}")

# Збереження результатів у CSV
results_df = pd.DataFrame(results)
results_df.to_csv('lab3_results.csv', index=False)
print(f"\nРезультати збережено у файл: lab3_results.csv")

print(f"\n" + "="*60)
print("Лабораторна робота №3 завершена успішно!")
print("Створено файли: lab3_plot.png, lab3_results.csv")
print("="*60)

# Показ графіка
plt.show()
