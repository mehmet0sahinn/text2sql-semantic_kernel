"""Natural Language to SQL with Semantic Kernel - Auto Function Invocation."""

import asyncio
import os
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from database_service import DatabaseService
from database_plugin import DatabasePlugin

load_dotenv()

SYSTEM_PROMPT = """You are an AI assistant that helps users query a SQL Server database using natural language.

You can:
1. Get database and schema information
2. List tables in a schema
3. Get column info for tables
4. Execute SQL SELECT queries

Rules:
- Only use SELECT statements, never modify data
- Use TOP 10 to limit results
- Explore schema first if unsure about structure
- Respond in user's language (Turkish/English)"""


def create_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    ))
    return kernel


def create_database_plugin() -> DatabasePlugin:
    return DatabasePlugin(DatabaseService(
        server=os.getenv("SQL_SERVER"),
        user=os.getenv("SQL_USER"),
        password=os.getenv("SQL_PASSWORD"),
        database=os.getenv("SQL_DATABASE")
    ))


async def chat(kernel: Kernel, history: ChatHistory, user_input: str) -> str:
    history.add_user_message(user_input)
    
    service: ChatCompletionClientBase = kernel.get_service(type=ChatCompletionClientBase)
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service.service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    response = await service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
        arguments=KernelArguments(settings=settings)
    )
    
    history.add_assistant_message(str(response))
    return str(response)


async def main():
    print("=" * 50)
    print("Natural Language to SQL")
    print("=" * 50)
    print("\nOrnek: 'SalesLT semasindaki tablolar neler?'")
    print("Cikis: 'exit' veya 'q'\n")

    kernel = create_kernel()
    kernel.add_plugin(create_database_plugin(), plugin_name="Database")
    
    history = ChatHistory()
    history.add_system_message(SYSTEM_PROMPT)

    while True:
        try:
            user_input = input("Siz: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", "cikis"]:
                print("Gorusmek uzere!")
                break
            
            print("AI: ", end="")
            print(await chat(kernel, history, user_input))
        except KeyboardInterrupt:
            print("\nGorusmek uzere!")
            break
        except Exception as e:
            print(f"Hata: {e}")


if __name__ == "__main__":
    asyncio.run(main())

