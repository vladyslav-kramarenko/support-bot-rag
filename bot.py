import os
import time
import html
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_chain import qa_chain

# === Load config ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SHOW_TECH_INFO = os.getenv("TECHNICAL_INFO", "false").lower() == "true"

# === HTML escaping ===
def escape_html(text: str) -> str:
    return html.escape(text)

# === Start command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I’m your Call Support Bot. Ask me any support-related question.")

# === Handle message ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    start_time = time.time()

    # Get response from RAG
    result = qa_chain.invoke({"query": user_input})
    elapsed = round(time.time() - start_time, 2)

    # Escape final answer body
    safe_result = escape_html(result['result'])

    # Prepare technical info if enabled
    technical_info = ""
    max_chunks_quantity = 5

    if SHOW_TECH_INFO:
        source_chunks = result.get("source_documents", [])
        if source_chunks:
            display_chunks = source_chunks[:max_chunks_quantity]
            chunk_blocks = []
            for i, chunk in enumerate(display_chunks):
                chunk_text = escape_html(chunk.page_content.strip()[:1000])
                chunk_blocks.append(f"<b>Chunk #{i+1}:</b>\n<pre>{chunk_text}</pre>")
            chunks_text = "\n\n".join(chunk_blocks)
        else:
            chunks_text = "<i>⚠️ No matching source chunks available.</i>"

        technical_info = f"""\
        <b>────────────────────────────</b>
<b>🛠️ Technical Info</b>

{chunks_text}

<b>⏱️ Response time:</b> {elapsed} seconds
"""

    # Final message
    final_reply = f"{safe_result}\n\n{technical_info}"

    # Truncate if too long
    MAX_MESSAGE_LENGTH = 4000
    if len(final_reply) > MAX_MESSAGE_LENGTH:
        final_reply = final_reply[:MAX_MESSAGE_LENGTH - 100] + "\n\n<i>⛔ Message truncated due to size</i>"

    await update.message.reply_text(final_reply.strip(), parse_mode="HTML")

# === Launch bot ===
def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot is running. Ask away on Telegram.")
    app.run_polling()

if __name__ == "__main__":
    run_bot()