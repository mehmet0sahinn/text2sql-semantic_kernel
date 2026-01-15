import os
import asyncio
import logging
import time
from dotenv import load_dotenv

from azure.monitor.opentelemetry import configure_azure_monitor

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory

from plugins.search_plugin import SearchPlugin


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    # Suppress Azure SDK logs
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    # Reduce OpenTelemetry noise
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    # Set Semantic Kernel to WARNING level
    logging.getLogger("semantic_kernel").setLevel(logging.WARNING)


def setup_observability():
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        raise ValueError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is missing.\n"
            "Fix: setx APPLICATIONINSIGHTS_CONNECTION_STRING \"InstrumentationKey=...;IngestionEndpoint=...\" \n"
            "or add it to your .env file."
        )
    configure_azure_monitor(connection_string=conn)


async def main():
    load_dotenv()

    setup_logging()
    setup_observability()

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

        start_time = time.time()
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=settings,
            kernel=kernel,
        )
        elapsed = time.time() - start_time

        print("Assistant >", str(result))
        
        # Show token and performance information
        metadata = result.metadata if hasattr(result, 'metadata') else {}
        usage = metadata.get('usage') if metadata else None
        
        info_parts = []
        if usage:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', 0)
            info_parts.append(f"{total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens})")
        
        info_parts.append(f"time: {elapsed:.2f}s")
        
        print(f"{' | '.join(info_parts)}\n")
        
        history.add_message(result)


if __name__ == "__main__":
    asyncio.run(main())