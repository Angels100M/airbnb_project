# 🏠 Airbnb Price Predictor & Telegram Bot

## תיאור הפרויקט
מערכת End-to-End לחיזוי מחירי לינה בדירות Airbnb בסן פרנסיסקו, עם:
- עיבוד וניקוי נתונים
- יצירת פיצ'רים (Feature Engineering)
- בניית מודל ML (XGBoost)
- UI אינטראקטיבי ב־Streamlit
- בוט טלגרם: קלט משתמש, חיזוי מחיר, קישור למפה, הסבר בעברית עם GPT, ושמירה של כל שיחה לדאטהבייס.

---
## 📁 מבנה תיקיות
airbnb_project/
│
├── app.py # קוד ה-Streamlit
├── requirements.txt # רשימת חבילות
├── README.md # מסמך זה
├── .env # משתני סביבה (סודי)
│
├── data/
│ ├── listings.csv # קובץ נתוני המקור
│ ├── cleaned_data.csv # נתונים לאחר ניקוי
│ ├── featured_data.csv # לאחר Feature Engineering
│ ├── X_train.csv # משתני קלט לאימון
│ └── chat_history.db # דאטהבייס לכל השיחות
│
├── models/
│ └── price_predictor_xgb.pkl # המודל המאומן
│
├── scripts/
│ ├── data_cleaning.py
│ ├── feature_engineering.py
│ ├── train_test_split.py
│ ├── train_model.py
│ └── telegram_bot.py
│
└── utils/
├── save_to_db.py
└── init.py

---
## ⚙️ דרישות מוקדמות

- Python 3.10 ומעלה (עדיף 3.11+)
- חבילת pip
- חשבון OpenAI API
- טוקן בוט טלגרם (BotFather)

---
## 🚀 התקנה והפעלה מהירה

1. **התקנת כל החבילות הדרושות:**
   ```sh
   pip install -r requirements.txt

הכנת קובץ .env בתיקיית הפרויקט:
TELEGRAM_BOT_TOKEN=your-telegram-token-here
OPENAI_API_KEY=sk-xxxxxx

הרצת שלבי עיבוד הנתונים (רוץ לפי הסדר):
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/train_test_split.py
python scripts/train_model.py

הרצת הממשק הגרפי :
streamlit run app.py

הרצת בוט הטלגרם:
python scripts/telegram_bot.py

## דוגמת קלט ופלט (בוט טלגרם)
קלט:
2 אורחים, 1 חדר שינה, 1 אמבטיה, Entire home/apt, Entire home, Downtown/Civic Center, 37.77, -122.42
פלט:
    מחיר משוער ללילה: $208
    קישור למפה ב-Google Maps
    הסבר אנושי בעברית + טיפ ללקוח (כולל אימוג'י), לדוגמה:
    1. השכונה Downtown/Civic Center נחשבת מרכזית ומעלה את המחיר.
2. סוג הנכס (דירה שלמה) מתאים במיוחד לזוגות או נוסעים בודדים.
3. מספר חדרי שינה ואורחים משפיעים משמעותית.
טיפ: כדאי לבדוק זמינות מחוץ לעונה כדי לקבל מחיר נמוך יותר! 💡


👨‍💻 מה למדתי ומה אתגרים עיקריים:
    עבודה מודולרית ונקייה עם מבנה תיקיות מקצועי
    טיפול בשגיאות ייבוא (PYTHONPATH, imports)
    עבודה מול גרסאות OpenAI חדשות (API)
    בניית מערכת שמשלבת ML, ממשק, API וחוויית משתמש בעברית
    שימוש ב־Git, כתיבת תיעוד, ושיתוף קוד נקי

    ## ובעיקר שרוב הזמן יוצא על תיקון תקלות שאני יוצר 🤣🤣🤣🤣

