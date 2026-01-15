import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory

from plugins.search_plugin import SearchPlugin


async def main():
    load_dotenv()

    kernel = Kernel()

    chat_completion = AzureChatCompletion(
        deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    kernel.add_service(chat_completion)

    # Search plugin
    kernel.add_plugin(SearchPlugin(), plugin_name="Search")

    settings = AzureChatPromptExecutionSettings()
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    history = ChatHistory()
    history.add_system_message(
        "You are a helpful assistant.\n"
        "- If the user asks something requiring external knowledge, call Search.search_docs first.\n"
        "- Use ONLY the SOURCES returned by Search. If not in sources, say you don't know.\n"
        "- When answering, cite sources like [1], [2]."
    )

    while True:
        user_input = input("User > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        history.add_user_message(user_input)

        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=settings,
            kernel=kernel,
        )

        print("Assistant >", str(result))
        history.add_message(result)


if __name__ == "__main__":
    asyncio.run(main())
