import os
import time
import html
import logging
import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_pipeline import qa_chain

# === Constants ===
MAX_MESSAGE_LENGTH = 4000
MAX_CHUNKS_TO_SHOW = 5

class HttpxFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ("HTTP/1.1 200 OK" in msg and "getUpdates" in msg)

# Apply the filter to httpx logger
httpx_logger = logging.getLogger("httpx")
httpx_logger.addFilter(HttpxFilter())

# === Load config ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# === Load YAML Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
SHOW_TECH_INFO = config.get("technical_info", False)

if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN is missing from .env file")

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

    # Escape final answer body safely
    safe_result = escape_html(result.get("result", "⚠️ No response generated."))

    # Prepare technical info if enabled
    technical_info = ""
    if SHOW_TECH_INFO:
        source_chunks = result.get("source_documents", [])
        if source_chunks:
            display_chunks = source_chunks[:MAX_CHUNKS_TO_SHOW]
            chunk_blocks = []
            for i, chunk in enumerate(display_chunks):
                chunk_text = escape_html(chunk.page_content.strip()[:1000])
                chunk_source = escape_html(chunk.metadata.get("source", "unknown"))
                chunk_blocks.append(
                    f"<b>Chunk #{i+1}:</b>\n<pre>{chunk_text}</pre>\n<i>Source: {chunk_source}</i>"
                )
            chunks_text = "\n\n".join(chunk_blocks)
        else:
            chunks_text = "<i>⚠️ No matching source chunks available.</i>"

        technical_info = f"""
<b>────────────────────────────</b>
<b>🛠️ Technical Info</b>

{chunks_text}

<b>⏱️ Response time:</b> {elapsed} seconds
"""

    # Final message
    final_reply = f"{safe_result}\n\n{technical_info}"

    # Truncate if too long
    if len(final_reply) > MAX_MESSAGE_LENGTH:
        await update.message.reply_text(
            escape_html("⚠️ Response too long to display. Please narrow your question."),
            parse_mode="HTML"
        )
        return

    await update.message.reply_text(final_reply.strip(), parse_mode="HTML")

# === Launch bot ===
def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Bot initialized. Waiting for messages...")
    app.run_polling()

if __name__ == "__main__":
    run_bot()
