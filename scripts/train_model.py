import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# טען את הנתונים
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# יצירת מודל XGBoost
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# אימון המודל
model.fit(X_train, y_train)

# חיזוי על סט הבדיקה
y_pred = model.predict(X_test)

# הערכת ביצועים
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# שמירת המודל המאומן
joblib.dump(model, 'models/price_predictor_xgb.pkl')
print("Trained model saved to models/price_predictor_xgb.pkl")


## שלב נוסף בניסיון להבין את הנתונים החשובים על מנת לשפר את המודל Feature Importance
import matplotlib.pyplot as plt

# שלוף את החשיבות מכל מאפיין (Feature)
importances = model.feature_importances_
feature_names = X_train.columns

# מיון לפי חשיבות
indices = importances.argsort()[::-1]
top_features = feature_names[indices][:10]
top_importances = importances[indices][:10]

# הדפסה טקסטואלית
print("\nTop 10 Important Features:")
for feature, importance in zip(top_features, top_importances):
    print(f"{feature}: {importance:.3f}")

# הדפסת גרף
plt.figure(figsize=(10,6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()
