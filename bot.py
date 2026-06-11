import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
    CommandHandler
)

TOKEN = "8873438835:AAFEy-mzaajNlMeF0zCqrsZ6Uc-dyvo_9J0"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

# Загружаем системный промпт один раз
try:
    with open("content.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except Exception as e:
    SYSTEM_PROMPT = (
        "Ты футбольный AI-аналитик. "
        "Не выдумывай факты. "
        "Если не знаешь — отвечай: нет данных."
    )
    print("Ошибка загрузки content.txt:", e)

keyboard = [
    ["📘 Правила футбола", "🧠 Тактики"],
    ["⚽ Игроки", "🏟 Команды"]
]

markup = ReplyKeyboardMarkup(
    keyboard,
    resize_keyboard=True
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я футбольный AI-ассистент.\n"
        "Задай вопрос или выбери тему ниже.",
        reply_markup=markup
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    if user_text == "📘 Правила футбола":
        user_text = "Объясни правила футбола простыми словами"

    elif user_text == "🧠 Тактики":
        user_text = "Объясни футбольные схемы 4-3-3, 4-4-2 и 3-5-2"

    elif user_text == "⚽ Игроки":
        user_text = "Назови известных футболистов и кратко расскажи о них"

    elif user_text == "🏟 Команды":
        user_text = "Назови лучшие футбольные клубы мира"

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False
    }

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=300
        )

        if response.status_code != 200:
            await update.message.reply_text(
                f"Ошибка LM Studio:\n{response.text}",
                reply_markup=markup
            )
            return

        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        # Ограничение Telegram
        if len(answer) > 4000:
            answer = answer[:4000]

        await update.message.reply_text(
            answer,
            reply_markup=markup
        )

    except Exception as e:
        await update.message.reply_text(
            f"Ошибка: {e}",
            reply_markup=markup
        )


def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Бот запущен")

    app.run_polling()


if __name__ == "__main__":
    main()
