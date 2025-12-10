import os
import csv
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import ta
import asyncio
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

# 1. SETUP & SECRETS
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID") or 0)
PRIVATE_GROUP_ID = int(os.getenv("PRIVATE_GROUP_ID") or 0) # Signals yahan jayenge
API_KEY = os.getenv("API_KEY")

# AI CONFIGURATION
LOOKBACK = 30
CONF_THRESHOLD = 0.65
TICKERS = [
    "EUR/USD", "GBP/USD", "USD/ZAR", 
    "BTC/USD", "ETH/USD", "XRP/USD", 
    "NVDA", "AAPL", "GOOG", "AMZN"
]

# LOGGING
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# CHAT STATES (Flow Management)
# 0, 1 used for basic flow. 2,3,4,5 for Verification
CHOOSING, WAITING_FOR_PERSONAL_MESSAGE = range(2)
ASK_NAME, ASK_ADDRESS, ASK_NUMBER, ASK_COUNTRY = range(2, 6)

reply_map = {}
active_personal_chats = set()
# Temporary storage for user verification data
user_data_cache = {}

# ==============================================================================
# 🧠 AI BRAIN SECTION
# ==============================================================================
print("🧠 Loading AI Models & Scalers...")
try:
    scaler_5m = joblib.load("models/scaler_5m.pkl")['scaler']
    scaler_15m = joblib.load("models/scaler_15m.pkl")['scaler']
    model_5m = tf.keras.models.load_model("models/model_5m.keras")
    model_15m = tf.keras.models.load_model("models/model_15m.keras")
    print("✅ AI Models Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Models not found. Error: {e}")
    model_5m = None

# ==============================================================================
# 📊 AI DATA FUNCTIONS
# ==============================================================================
def add_features(df):
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = (sma20 + 2 * std20 - df['close']) / df['close']
    df['bb_lower'] = (df['close'] - (sma20 - 2 * std20)) / df['close']
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-8)
    else:
        df['vol_ma'] = 0
    ema50 = df['close'].ewm(span=50).mean()
    df['dist_ema50'] = (df['close'] - ema50) / ema50
    return df.dropna()

def fetch_live_data(symbol, interval):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": 60, "apikey": API_KEY}
    try:
        import requests
        resp = requests.get(url, params=params).json()
        if "values" not in resp: return None
        df = pd.DataFrame(resp["values"])
        df = df.iloc[::-1].reset_index(drop=True)
        cols = ["open", "high", "low", "close"]
        if "volume" in df.columns: cols.append("volume")
        df[cols] = df[cols].astype(float)
        df = add_features(df)
        if len(df) < LOOKBACK: return None
        return df.tail(LOOKBACK)
    except Exception as e:
        logger.error(f"Data error {symbol}: {e}")
        return None

def prepare_input(df, scaler):
    features = ['log_ret', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'vol_ma', 'dist_ema50']
    for col in features:
        if col not in df.columns: df[col] = 0.0
    data = df[features].values
    scaled = scaler.transform(data)
    return np.array([scaled])

# ==============================================================================
# 🤖 AI SCANNER JOB (Sends Signals to PRIVATE GROUP)
# ==============================================================================
async def ai_scanner_job(context: ContextTypes.DEFAULT_TYPE):
    if model_5m is None: return
    
    # IMPORTANT: Signals now go to Private Group, not Owner
    TARGET_CHAT = PRIVATE_GROUP_ID if PRIVATE_GROUP_ID != 0 else OWNER_CHAT_ID
    
    print(f"🔍 Scanning Market... (Target: {TARGET_CHAT})")

    for symbol in TICKERS:
        df_5m = fetch_live_data(symbol, "5m")
        if df_5m is not None:
            x = prepare_input(df_5m, scaler_5m)
            prob = model_5m.predict(x, verbose=0)[0][0]
            price = df_5m['close'].iloc[-1]
            
            if prob > CONF_THRESHOLD:
                signal_msg = (
                    f"🚀 <b>PREMIUM SIGNAL</b>\n"
                    f"💎 Symbol: <b>{symbol}</b>\n"
                    f"⏱️ Timeframe: 5 Min\n"
                    f"🧠 Confidence: {prob*100:.1f}%\n"
                    f"💲 Price: {price}\n"
                    f"🎯 Action: <b>BUY NOW</b>"
                )
                await context.bot.send_message(chat_id=TARGET_CHAT, text=signal_msg, parse_mode="HTML")
        await asyncio.sleep(2)

# ==============================================================================
# 📝 VERIFICATION FLOW (Name -> Address -> Number -> Country -> Add)
# ==============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("🗣️ Talk to Owner", callback_data="owner")],
        [InlineKeyboardButton("📢 Group / Channel Help", callback_data="group_help")],
    ]
    await update.message.reply_text(
        "👋 Welcome to Stock AI Bot!\nChoose an option:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return CHOOSING

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    choice = query.data
    
    if choice == "owner":
        active_personal_chats.add(query.from_user.id)
        await query.edit_message_text("💬 *Owner Chat Active*\nType your message below.", parse_mode="Markdown")
        return WAITING_FOR_PERSONAL_MESSAGE

    if choice == "group_help":
        keyboard = [
            [InlineKeyboardButton("🌐 Public Group", url="https://t.me/YOUR_PUBLIC_LINK")],
            [InlineKeyboardButton("📢 Channel", url="https://t.me/YOUR_CHANNEL_LINK")],
            [InlineKeyboardButton("🔒 Request Private Group", callback_data="start_verification")],
        ]
        await query.edit_message_text(
            "📂 *Group Directory*\nSelect where you want to go:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return CHOOSING # Stay in choosing state to handle next click

    if choice == "start_verification":
        await query.edit_message_text(
            "🔒 *Private Group Verification*\n\nTo join our Premium Signals group, we need some details.\n\n1️⃣ **What is your Full Name?**",
            parse_mode="Markdown"
        )
        return ASK_NAME

    return ConversationHandler.END

# --- Verification Steps ---

async def get_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_data_cache[update.message.from_user.id] = {'name': update.message.text}
    await update.message.reply_text("2️⃣ **Thanks! Now, what is your Address?**", parse_mode="Markdown")
    return ASK_ADDRESS

async def get_address(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_data_cache[update.message.from_user.id]['address'] = update.message.text
    await update.message.reply_text("3️⃣ **Please share your Phone Number:**", parse_mode="Markdown")
    return ASK_NUMBER

async def get_number(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_data_cache[update.message.from_user.id]['number'] = update.message.text
    await update.message.reply_text("4️⃣ **Last step: Which Country are you from?**", parse_mode="Markdown")
    return ASK_COUNTRY

async def finish_verification(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    user_data = user_data_cache.get(user_id)
    user_data['country'] = update.message.text # Save country
    
    # 1. Save Data to CSV
    file_exists = os.path.isfile('verified_users.csv')
    with open('verified_users.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(['User ID', 'Name', 'Address', 'Number', 'Country'])
        writer.writerow([user_id, user_data['name'], user_data['address'], user_data['number'], user_data['country']])

    # 2. Generate Single-Use Invite Link for Private Group
    try:
        invite_link = await context.bot.create_chat_invite_link(
            chat_id=PRIVATE_GROUP_ID,
            member_limit=1, # One link per user
            name=f"Invite for {user_data['name']}"
        )
        link_text = invite_link.invite_link
    except Exception as e:
        link_text = "Error generating link. Owner will add you manually."
        logger.error(f"Link Gen Error: {e}")

    # 3. Notify User
    await update.message.reply_text(
        f"✅ **Verification Complete!**\n\nHere is your unique link to join the Private Signals Group:\n{link_text}",
        parse_mode="Markdown"
    )

    # 4. Notify Owner
    owner_msg = (
        f"👤 **New User Verified!**\n"
        f"Name: {user_data['name']}\n"
        f"Phone: {user_data['number']}\n"
        f"From: {user_data['country']}\n"
        f"ID: `{user_id}`"
    )
    await context.bot.send_message(chat_id=OWNER_CHAT_ID, text=owner_msg, parse_mode="Markdown")
    
    return ConversationHandler.END

# --- Message Forwarding (Existing Logic) ---

async def receive_personal_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    if user.id not in active_personal_chats:
        return ConversationHandler.END # Ignore unexpected text

    if update.message.text.lower() == '/cancel':
        active_personal_chats.discard(user.id)
        await update.message.reply_text("Exited owner chat.")
        return ConversationHandler.END

    # Forward to Owner
    fwd = f"📩 *User Msg* ({user.full_name}):\n{update.message.text}"
    sent = await context.bot.send_message(chat_id=OWNER_CHAT_ID, text=fwd, parse_mode="Markdown")
    reply_map[sent.message_id] = user.id
    await update.message.reply_text("✅ Sent.")
    return WAITING_FOR_PERSONAL_MESSAGE

async def owner_reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.from_user.id != OWNER_CHAT_ID: return
    if update.message.reply_to_message:
        tid = reply_map.get(update.message.reply_to_message.message_id)
        if tid:
            await context.bot.send_message(chat_id=tid, text=f"📬 *Owner:* {update.message.text}", parse_mode="Markdown")
            await update.message.reply_text("✅ Replied.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    active_personal_chats.discard(update.message.from_user.id)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Conversation Handler with Verification States
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING: [CallbackQueryHandler(button_handler)],
            WAITING_FOR_PERSONAL_MESSAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_personal_message)],
            
            # New Verification States
            ASK_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_name)],
            ASK_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_address)],
            ASK_NUMBER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_number)],
            ASK_COUNTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, finish_verification)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & filters.User(OWNER_CHAT_ID), owner_reply_handler))

    if app.job_queue:
        app.job_queue.run_repeating(ai_scanner_job, interval=300, first=10)
        print("✅ AI Scanner Active")

    print("🤖 Bot Started...")
    app.run_polling()


if __name__ == "__main__":
    main()