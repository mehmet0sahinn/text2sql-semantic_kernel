import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenAI SDK client
client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url="https://ai-mehmetaihub198370024490.openai.azure.com/openai/v1/",
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

def ask(q: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        print("AI:", ask(q))
