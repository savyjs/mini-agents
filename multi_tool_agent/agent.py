from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm

# --- Example Agent using OpenAI's GPT-4o ---
# (Requires OPENAI_API_KEY)
root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),  # LiteLLM model string format

    name="weather_time_agent",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
)
