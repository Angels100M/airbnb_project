import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from utils.save_to_db import save_interaction
import dotenv
dotenv.load_dotenv()
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# טען את המודל
MODEL_PATH = os.path.join("models", "price_predictor_xgb.pkl")
FEATURE_COLUMNS_PATH = os.path.join("data", "X_train.csv")

model = joblib.load(MODEL_PATH)
feature_columns = pd.read_csv(FEATURE_COLUMNS_PATH).columns.tolist()

# דוגמאות לערכים
property_types = sorted([c.replace('property_type_', '') for c in feature_columns if c.startswith('property_type_')])
room_types = sorted([c.replace('room_type_', '') for c in feature_columns if c.startswith('room_type_')])
neighbourhoods = sorted([c.replace('neighbourhood_cleansed_', '') for c in feature_columns if c.startswith('neighbourhood_cleansed_')])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "שלום! שלח לי פרטים על דירה (פורמט: מספר אורחים, חדרי שינה, אמבטיות, סוג חדר, סוג נכס, שכונה, קו''ר ולונג')\n"
        "לדוגמה:\n"
        "2 אורחים, 1 חדר שינה, 1 אמבטיה, Entire home/apt, Entire home, Downtown/Civic Center, 37.77, -122.42"
    )

def parse_message(text):
    # דוגמה לפרסור: !
    try:
        parts = [p.strip() for p in text.split(',')]
        accommodates = int(parts[0].split()[0])
        bedrooms = float(parts[1].split()[0])
        bathrooms = float(parts[2].split()[0])
        room_type = parts[3]
        property_type = parts[4]
        neighbourhood = parts[5]
        latitude = float(parts[6])
        longitude = float(parts[7])
        return {
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'room_type': room_type,
            'property_type': property_type,
            'neighbourhood_cleansed': neighbourhood,
            'latitude': latitude,
            'longitude': longitude,
            'beds': bedrooms  # הערכה — אפשר לשפר בהמשך
        }
    except Exception as e:
        return None

## הוספת פונקצית הסבר של GPT
def gpt_explanation(features, pred):
    prompt = (
        "אתה יועץ נדל\"ן ידידותי ומקצועי ב-Airbnb בסן פרנסיסקו. "
        "קיבלת נתונים של דירה לצורך הערכת מחיר. "
        "הסבר בעברית בקצרה, בטון אישי, מהם הגורמים העיקריים למחיר לפי הנתונים (התייחס לסוג הדירה, סוג החדר, מספר האורחים, חדרי שינה, חדרי אמבטיה והשכונה). "
        "הוסף בסוף כל הסבר טיפ קטן להוזלת המחיר או להשגת עסקה טובה, וסיים את ההסבר עם אימוג'י מתאים שמסמל בית, כסף או רעיון. "
        "דאג שהתשובה תכלול לפחות שלוש נקודות בולטות (1,2,3) ותהיה בגובה העיניים. "
        f"\nנתוני הדירה:\n"
        f"שכונה: {features['neighbourhood_cleansed']}\n"
        f"סוג נכס: {features['property_type']}\n"
        f"סוג חדר: {features['room_type']}\n"
        f"חדרי שינה: {features['bedrooms']}\n"
        f"חדרי אמבטיה: {features['bathrooms']}\n"
        f"מיטות: {features['beds']}\n"
        f"אורחים: {features['accommodates']}\n"
        f"מחיר משוער ללילה: ${pred:,.0f}\n"
        "הסבר אישי + טיפ להוזלת מחיר:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=220,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


async def predict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message.text
    features = parse_message(msg)
    if features is None:
        await update.message.reply_text("הקלט לא מזוהה. נסה שוב בפורמט המתואר עם פסיקים בין שדות.")
        return

    # בניית וקטור מלא
    input_dict = {key: features[key] for key in ['latitude','longitude','accommodates','bathrooms','bedrooms','beds']}
    for p in property_types:
        input_dict[f'property_type_{p}'] = 1 if p == features['property_type'] else 0
    for r in room_types:
        input_dict[f'room_type_{r}'] = 1 if r == features['room_type'] else 0
    for n in neighbourhoods:
        input_dict[f'neighbourhood_cleansed_{n}'] = 1 if n == features['neighbourhood_cleansed'] else 0
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0

    X_input = pd.DataFrame([input_dict], columns=feature_columns)
    pred = model.predict(X_input)[0]
    await update.message.reply_text(f"❗ מחיר משוער ללילה: ${pred:,.0f}")

    # החזרת קישור ל־Google Maps לפי קו״ר ולונג
    maps_url = f"https://www.google.com/maps/search/?api=1&query={features['latitude']},{features['longitude']}"
    await update.message.reply_text(f"מיקום הדירה במפה: {maps_url}")

    # החזרת הסבר GPT
    explanation = gpt_explanation(features, pred)
    await update.message.reply_text(f"הסבר:\n{explanation}")

    # שמירה לדאטהבייס
    user_id = str(update.effective_user.id)
    user_message = msg
    prediction = f"${pred:,.0f}"
    gpt_response = explanation
    save_interaction(user_id, user_message, prediction, gpt_response)



if __name__ == '__main__':
    
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  
    if TOKEN is None:
        print("שים את הטוקן שלך בקובץ .env בצורה TELEGRAM_BOT_TOKEN=xxxx")
        exit(1)

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), predict_handler))

    print("הבוט רץ... שלח הודעה בטלגרם לבוט שלך!")
    app.run_polling()


