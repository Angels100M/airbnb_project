import pandas as pd

# ===========================
# שלב 1: טעינת הנתונים
# ===========================
csv_path = 'data/listings.csv'
df = pd.read_csv(csv_path)

# ניקוי שמות כל העמודות מרווחים ותווים בלתי נראים
# ניקוי שמות עמודות
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('\u200e', '')  # תווים נסתרים
df.columns = df.columns.str.lower()

print("All columns in raw data:")
for col in df.columns:
    print(f"'{col}'")

# ===========================
# שלב 2: בחירת עמודות חשובות בלבד (כולל neighborhood_cleansed)
# ===========================
columns_to_keep = [
    'latitude', 'longitude', 'neighbourhood_cleansed',
    'property_type', 'room_type',
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price'
]
missing = [col for col in columns_to_keep if col not in df.columns]
print("Missing columns:", missing)  # חשוב: תראה כאן בדיוק איזה עמודות חסרות!

df = df[columns_to_keep]

print("\nFirst 5 rows after filtering:")
print(df.head())

# ===========================
# שלב 3: מילוי ערכים חסרים ב-neighborhood_cleansed
# ===========================
df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].fillna('Unknown')
print("\nNeighborhood missing values filled with 'Unknown'.")

# ===========================
# שלב 4: המרת עמודת price למספר (float)
# ===========================
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
print("\nPrice column after conversion to float:")
print(df['price'].head())

# ===========================
# שלב 5: הסרת שורות עם ערכים חסרים בעמודות קריטיות
# *לא מסירים לפי neighborhood_cleansed!*
# ===========================
critical_columns = [
    'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'price'
]
df = df.dropna(subset=critical_columns)
print(f"\nShape after dropping rows with missing critical values: {df.shape}")

# ===========================
# שלב 6: טיפול בערכים חסרים בעמודות משניות (מילוי חציון)
# ===========================
for col in ['bathrooms', 'bedrooms', 'beds']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
print("\nMissing values after median fill:")
print(df.isnull().sum())

# ===========================
# שלב 7: סינון חריגים במחיר
# ===========================
df = df[(df['price'] >= 40) & (df['price'] <= 1000)]
print(f"\nAfter outlier filtering, data shape: {df.shape}")

# ===========================
# שלב 8: בדיקת neighborhood_cleansed לפני שמירה
# ===========================
print("\nValue counts for neighborhood_cleansed:")
print(df['neighbourhood_cleansed'].value_counts())

# ===========================
# שלב 9: שמירת הקובץ הנקי
# ===========================
df.to_csv('data/cleaned_data.csv', index=False)
print("\nCleaned data saved to data/cleaned_data.csv")
