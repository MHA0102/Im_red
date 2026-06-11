import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

tb_data = pd.read_csv('TB_Burden_Country.csv')

y = tb_data['Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population']

features = ['Year', 'Estimated total population number',
            'Estimated prevalence of TB (all forms) per 100 000 population',
            'Estimated incidence (all forms) per 100 000 population']

X = tb_data[features].dropna()
y = y[X.index]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

mae = mean_absolute_error(val_y, model.predict(val_X))

print("=" * 40)
print("  Модель: Дерево решений")
print("  Датасет: Туберкулёз по странам")
print("=" * 40)
print(f"  Обучающая выборка:    {train_X.shape[0]} строк")
print(f"  Валидационная выборка: {val_X.shape[0]} строк")
print("-" * 40)
print(f"  MAE: {mae:.2f} смертей на 100 000")
print("=" * 40)