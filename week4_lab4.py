from dotenv import load_dotenv
import os
import requests
from langchain_core.tools import Tool

load_dotenv(override=True)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(text: str):
    
    payload = {
        "user": pushover_user,
        "token": pushover_token,
        "message": text
    }

    requests.post(
        pushover_url,
        data=payload
    )

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Useful for when you want to send a push notification"
)

tool_push.invoke("Hello, me!")




