import logging


def log_conversation(ai_response, user_question, is_new_upload):
    prefix = "[New File Upload]" if is_new_upload else "[Continued Conversation]"
    log_entry = f"{prefix} AI: {ai_response} User: {user_question}\n"
    try:
        with open("conversations.log", "a") as log_file:
            log_file.write(log_entry)
    except IOError as e:
        logging.error(f"Failed to log conversation: {e}")
