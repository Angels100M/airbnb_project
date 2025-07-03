import pandas as pd

# טען את הנתונים הנקיים
df = pd.read_csv('data/cleaned_data.csv')

# הגדרת עמודות קטגוריות, רק אם הן באמת קיימות
categorical_columns = [col for col in ['property_type', 'room_type', 'neighbourhood_cleansed'] if col in df.columns]

# קידוד One Hot
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# שמירת הנתונים המקודדים
df_encoded.to_csv('data/featured_data.csv', index=False)

print("Feature engineering done! Saved to data/featured_data.csv")
print("New columns:")
print(df_encoded.columns)
print("Shape of featured data:", df_encoded.shape)
