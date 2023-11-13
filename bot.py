import json
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)

from src import model_selection as ms

with open("private/api_codes.json", "r") as json_file:
    TOKEN = json.load(json_file)["telegram"]

print("Started ✅")

folder = "private/storage/docs/2023-11-01_EN_meta/"


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("hey")


def answer(update: Update, context: CallbackContext) -> None:
    ans = ms.model1107meta(update.message.text, folder)
    update.message.reply_text(ans)


def main() -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, answer))
    updater.start_polling()
    updater.idle()


main()
