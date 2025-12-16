# =========== ЛАБОРАТОРНА РОБОТА №1 ===========
# Полтавський національний технічний університет ім. Юрія Кондратюка
# Студентка: Тугаріна Емілія, група 304-тн
# Варіант: 23

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 70)
print("ПОЛТАВСЬКИЙ НАЦІОНАЛЬНИЙ ТЕХНІЧНИЙ УНІВЕРСИТЕТ")
print("імені Юрія Кондратюка")
print("Лабораторна робота №1: Лінійна регресія")
print("Виконала: Тугарина Емілія, група 304-тн")
print("Варіант: 23")
print("=" * 70)

# ========== 1. ГЕНЕРАЦІЯ ДАНИХ ==========
n = 23
np.random.seed(42)

x = np.linspace(0, 10, 100)
y = n * x + np.sin(x / n) + np.random.normal(0, 1, 100)

print("\n1. 📊 ГЕНЕРАЦІЯ ДАНИХ:")
print(f"   • Згенеровано 100 точок")
print(f"   • Формула: y = {n}*x + sin(x/{n}) + шум")
print(f"   • Діапазон x: від {min(x):.1f} до {max(x):.1f}")
print(f"   • Діапазон y: від {min(y):.1f} до {max(y):.1f}")

# ========== 2. РОЗДІЛЕННЯ ДАНИХ ==========
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print(f"\n2. 📈 РОЗДІЛЕННЯ ДАНИХ:")
print(f"   • Навчальна вибірка: {len(x_train)} точок (70%)")
print(f"   • Тестова вибірка: {len(x_test)} точок (30%)")

# ========== 3. ПОБУДОВА МОДЕЛІ ==========
model = LinearRegression()
model.fit(x_train.reshape(-1, 1), y_train)

a = model.coef_[0]
b = model.intercept_

print(f"\n3. 🎯 РЕЗУЛЬТАТИ МОДЕЛІ:")
print(f"   • Коефіцієнт нахилу (a) = {a:.6f}")
print(f"   • Точка перетину (b) = {b:.6f}")
print(f"   • Рівняння моделі: y = {a:.4f}·x + {b:.4f}")

# ========== 4. ОЦІНКА ЯКОСТІ ==========
y_pred = model.predict(x_test.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n4. 📐 МЕТРИКИ ЯКОСТІ:")
print(f"   • Середньоквадратична помилка (MSE) = {mse:.6f}")
print(f"   • Середня абсолютна помилка (MAE) = {mae:.6f}")
print(f"   • Коефіцієнт детермінації (R²) = {r2:.6f}")

# ========== 5. ГРАФІК ==========
plt.figure(figsize=(12, 8))

# Точки даних
plt.scatter(x_train, y_train, color='lightblue', alpha=0.6, 
           s=40, label='Навчальна вибірка (70%)')
plt.scatter(x_test, y_test, color='blue', alpha=0.7, 
           s=50, label='Тестова вибірка (30%)')

# Лінія регресії
x_line = np.linspace(min(x), max(x), 300)
y_line = a * x_line + b
plt.plot(x_line, y_line, color='red', linewidth=3, 
        label=f'Лінія регресії: y = {a:.2f}x + {b:.2f}')

# Налаштування графіка
plt.xlabel('Значення x', fontsize=12, fontweight='bold')
plt.ylabel('Значення y', fontsize=12, fontweight='bold')
plt.title(f'ЛІНІЙНА РЕГРЕСІЯ - Варіант {n}\nТугарина Емілія, група 304-тн', 
          fontsize=14, fontweight='bold', pad=20)

# Інформаційна панель
textstr = f'РЕЗУЛЬТАТИ:\nMSE = {mse:.3f}\nMAE = {mae:.3f}\nR² = {r2:.3f}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Збереження графіка
filename = f'lab1_variant_{n}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\n5. 📸 ГРАФІК:")
print(f"   • Збережено як: {filename}")

# Показ графіка
plt.show()

# ========== 6. РЕЗЮМЕ ==========
print("\n" + "=" * 70)
print("📋 РЕЗЮМЕ ДЛЯ ЗВІТУ:")
print("=" * 70)
print("\nТАБЛИЦЯ РЕЗУЛЬТАТІВ:")
print("| Параметр           | Значення       |")
print("|--------------------|----------------|")
print(f"| Номер варіанта     | {n:14} |")
print(f"| Коефіцієнт a       | {a:14.4f} |")
print(f"| Коефіцієнт b       | {b:14.4f} |")
print(f"| MSE               | {mse:14.4f} |")
print(f"| MAE               | {mae:14.4f} |")
print(f"| R²                | {r2:14.4f} |")

print(f"\nРІВНЯННЯ МОДЕЛІ:")
print(f"y = {a:.4f}·x + {b:.4f}")

print(f"\n📊 ВИСНОВКИ:")
print(f"1. Модель: y = {a:.2f}x + {b:.2f}")
print(f"2. Якість моделі (R²): {r2:.3f} - {'ВИСОКА' if r2 > 0.9 else 'СЕРЕДНЯ' if r2 > 0.7 else 'НИЗЬКА'}")
print(f"3. Середня помилка (MAE): {mae:.3f}")
print(f"4. Модель {'добре' if r2 > 0.9 else 'задовільно' if r2 > 0.7 else 'погано'} описує дані")
print("=" * 70)

# Запис результатів у файл
with open('results.txt', 'w', encoding='utf-8') as f:
    f.write("РЕЗУЛЬТАТИ ЛАБОРАТОРНОЇ РОБОТИ №1\n")
    f.write("="*50 + "\n")
    f.write(f"Варіант: {n}\n")
    f.write(f"Рівняння: y = {a:.4f}x + {b:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write("="*50)

print(f"\n✅ Результати також збережено у файлі: results.txt")
print("🎉 Лабораторну виконано успішно!")
