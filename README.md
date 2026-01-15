# AI Developer - Building Intelligent Apps Hackathon

This repository contains projects developed for the Microsoft AI Developer Hackathon. Each challenge folder focuses on a different AI integration and scenario.

---

## Challenge 3 - Azure OpenAI Chat Applications

Four different approaches to building an Azure OpenAI chat application:

**app1.py**: Basic chatbot using the OpenAI SDK. Connects to Azure OpenAI using the vanilla OpenAI client.

**app2.py**: Chat application using the Azure OpenAI SDK. Uses the `AzureOpenAI` client for Azure-specific authentication and endpoint management.

**app3.py**: RAG application using Azure OpenAI "On Your Data" feature. Integrates with Azure AI Search for hybrid (vector + keyword) search, retrieves documents and injects them as context to the LLM. Uses Azure-side retrieval.

**app4.py**: Manual RAG implementation. Manually creates embeddings and performs hybrid search from Azure AI Search, then sends results as context to the LLM. Full control remains with the developer.

---

## Challenge 4 - Plugin-Based RAG with Semantic Kernel

RAG application developed using the Semantic Kernel framework with plugin architecture.

- `SearchPlugin`: Plugin that performs hybrid search with Azure AI Search. Handles embedding creation, vector search, and result formatting.
- `LightsPlugin`: Sample actuator plugin. Provides functions to read and change light states.
- `FunctionChoiceBehavior.Auto()` allows the LLM to automatically determine which plugin to call.
- Chat history provides conversation memory.

---

## Challenge 5 - Observability and Azure Monitor Integration

Added observability with Azure Monitor on top of Challenge 4.

- `configure_azure_monitor()` sends telemetry to Application Insights.
- Distributed tracing with OpenTelemetry.
- Token usage and response time are measured and displayed to the user.
- Logging levels configured to filter unnecessary Azure SDK logs.

---

## Challenge 6 - RAG with Cosmos DB

Vector search-based RAG application using Azure Cosmos DB NoSQL.

- `ingest_recipes_to_cosmos.py`: Loads JSONL data into Cosmos DB and generates embedding vectors.
- Retrieves relevant documents from Cosmos DB using vector search.
- Uses `VectorDistance` function for similarity ranking.
- Observability with OpenTelemetry.
- Conversation memory supports follow-up questions.

---

## Challenge 7 - Natural Language to SQL (NL2SQL)

Agent that converts natural language questions into T-SQL queries.

- Uses Semantic Kernel `ChatCompletionAgent`.
- Database schema from `dbschema.txt` is provided to the LLM.
- Security measures block SQL injection and dangerous queries (INSERT, UPDATE, DELETE, DROP, etc. are prohibited).
- Streaming response for real-time answer display.
- Only SELECT and CTE (WITH) queries are allowed.

---

## Challenge 8 - NL2SQL with Auto Function Calling

Fully automated NL2SQL application using Semantic Kernel plugin architecture.

**DatabaseService**: Establishes ODBC connection with SQL Server. Retrieves schema, table, column information and executes SQL queries.

**DatabasePlugin**: Exposes database operations as a Semantic Kernel plugin:
- `get_database_info`: Database information
- `get_database_schema_info`: Schema list
- `get_database_schema_table_info`: Tables
- `get_database_schema_table_columns_info`: Column information
- `execute_sql_command`: Execute SQL query (SELECT only)

With `FunctionChoiceBehavior.Auto()`, the LLM can automatically choose which function to call and when. The user asks questions in natural language, the LLM explores the schema, and executes the appropriate SQL.

---

## Techn

- Azure OpenAI (GPT-4, Embeddings)
- Azure AI Search (Vector + Hybrid Search)
- Azure Cosmos DB NoSQL (Vector Search)
- Semantic Kernel (Python)
- Azure Monitor / Application Insights
- OpenTelemetry
- SQL Server (pyodbc)
- Python 3.x
