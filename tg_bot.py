import os
import uuid
import json
import requests
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configuration
BOT_TOKEN = os.getenv('NF_HW_BOT_TOKEN')  # Set your bot token as environment variable
API_URL = "http://localhost:8000/process-string"  # Replace with your actual API URL

# Store user sessions in memory (in production, use a database)
user_sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    user_id = update.effective_user.id
    await create_new_session(user_id, update, context)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /reset command"""
    user_id = update.effective_user.id
    await create_new_session(user_id, update, context)

async def create_new_session(user_id: int, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Create a new session UUID for user and send welcome message"""
    # Generate new UUID for the session
    new_session_id = str(uuid.uuid4())
    user_sessions[user_id] = new_session_id
    
    # Create keyboard with options
    keyboard = [["/start", "/reset"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    # Send welcome message
    welcome_text = (
        "🔄 Сеанс обновлён!\n\n"
        f"Ваш новый ID сеанса: `{new_session_id}`\n\n"
        "Задавайте вопросы, либо введите /start or /reset для обновления сеанса ещё раз."
    )
    
    await update.message.reply_text(
        welcome_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages from users"""
    user_id = update.effective_user.id
    
    # Get or create session ID for user
    session_id = user_sessions.get(user_id)
    if not session_id:
        session_id = str(uuid.uuid4())
        user_sessions[user_id] = session_id
    
    user_question = update.message.text
    
    # Create the request dictionary
    request_data = {
        "session_id": session_id,
        "question": user_question
    }
    
    try:
        # Send typing action to show bot is working
        await update.message.chat.send_action(action="typing")
        
        # Send HTTP request to the API
        response = requests.post(
            API_URL,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=180
        )
        response.raise_for_status()
        
        # Parse response
        api_response = response.json()
        
        print(f"Response received: {api_response}")
        
        # Extract answer and source documents
        answer = api_response.get("answer", "[нет ответа]")
        source_documents = api_response.get("source_documents", [])
        
        # Format the response
        if not source_documents:
            # If no source documents, just send the answer
            response_text = answer
        else:
            # If there are source documents, append each source
            response_text = answer + "\n\n**Источники:**\n"
            for doc in source_documents:
                source = doc.get("source", "[неизвестный источник]")
                response_text += f"\n{source}"
        
        # Send the response back to user
        await update.message.reply_text(
            response_text,
            #parse_mode=telegram.constants.ParseMode.MARKDOWN_V2,
            reply_markup=ReplyKeyboardMarkup([["/start", "/reset"]], resize_keyboard=True)
        )
        
    except requests.exceptions.RequestException as e:
        error_message = f"❌ Ошибка при обработке HTTP-запроса: {str(e)}"
        await update.message.reply_text(error_message)
    except json.JSONDecodeError:
        error_message = "❌ Сервер сообщает об ошибке."
        await update.message.reply_text(error_message)
    except Exception as e:
        error_message = f"❌ Неожиданная ошибка: {str(e)}"
        await update.message.reply_text(error_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command"""
    help_text = (
        "🤖 **Помощь**\n\n"
        "**Команды:**\n"
        "• /start - Начать новый сеанс UUID\n"
        "• /reset - Рестарт сеанса UUID\n\n"
        "**Использование:**\n"
        "• Задайте вопрос по любой теме, а также можете спросить что-нибудь про Neoflex.\n"
        "• Можете использовать команды /start или /reset чтобы начать диалог заново."
    )
    
    await update.message.reply_text(
        help_text,
        parse_mode='Markdown',
        reply_markup=ReplyKeyboardMarkup([["/start", "/reset"]], resize_keyboard=True)
    )

def main():
    """Start the bot"""
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()