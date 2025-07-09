from google.adk.agents import LlmAgent, Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import google_search, agent_tool

# --- Example Agent using OpenAI's GPT-4o ---

search_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='SearchAgent',
    instruction="""
    You're a specialist in Google Search
    """,
    tools=[google_search],
)

# (Requires OPENAI_API_KEY)
root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),  # LiteLLM model string format

    name="weather_time_agent",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the coding using google search"
    ),
    tools=[agent_tool.AgentTool(agent=search_agent)],
)
