## חלוקת הנתונים לאימון ובדיקה

import pandas as pd
from sklearn.model_selection import train_test_split

# טען את הקובץ המהונדס
df = pd.read_csv('data/featured_data.csv')

# משתני קלט (כל העמודות למעט price)
X = df.drop('price', axis=1)
# עמודת מחיר (המשתנה לחיזוי)
y = df['price']

# חלוקה ל־train ו־test (80% לאימון, 20% לבדיקה)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# שמירה לקבצים חדשים
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Train/Test split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
