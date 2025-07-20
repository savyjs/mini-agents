from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import google_search, agent_tool

# --- Example Agent using OpenAI's GPT-4o ---

human_authenticity_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='AuthenticityAgent',
    instruction="You're a specialist in rephrasing in a way that seems like human made content. don't use AI pattern in writing. don't use - as comma, keep text same as Ehsan provided prompt, and make sure it is slightly more professional.  Ehsan is a 28 years old software developer who works in E-Commerce industry and knew TypeScript + PHP + MySql + Kafka - He want to build and sell SaaS software, specially in E-Commerce or Cryptocurrency niches or anything that could be sold easily online"
)

english_grammar_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='EnglishGrammarAgent',
    instruction="You're a specialist in Fixing English Grammar - score user grammar naturality an score between 0 - 100 - and provide response in two categories: 1- slightly changed grammar 2- fully rephrased grammar",
    tools=[agent_tool.AgentTool(agent=human_authenticity_agent)]
)

social_media_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='SocialMediaAgent',
    instruction="you are senior social media content creator - you know how to shape contents, making it visible, marketing through content and using hooks to increase impressions"
)

graphic_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='GraphicAndMediaAgent',
    instruction="you are senior graphist - you can choose and offer best possible ideas for covers and between text paragraphs of a content"
)

business_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),
    name='BusinessAgent',
    instruction="You're a specialist in  Business - you are experienced e-commerce manager"
)

# (Requires OPENAI_API_KEY)
root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o"),  # LiteLLM model string format

    name="contentStrategist",
    description=(
        "Agent to help Ehsan (user) write his articles."
    ),
    instruction=(
        "You are a helpful agent who can help ehsan write his articles. you can manage tools in order to create an amazing content."
        "you decide using tools in order to provide Ehsan with final version of his article that He can share on social media."
        "first section of reply: first check your tools and decide how to use them."
        "second section of reply: mention which sections are good and which sections need to be changed and which section is not necessary or should be added"
        "third section: score original content and highlight top 5 of worst phrases, if every thing is good mention"
        "fourth section: provide final content that Ehsan can copy easily"
        "5th section: if there is a suggestion or feedback mention that"
    ),
    tools=[agent_tool.AgentTool(agent=social_media_agent), agent_tool.AgentTool(agent=graphic_agent),
           agent_tool.AgentTool(agent=business_agent), agent_tool.AgentTool(agent=english_grammar_agent)],
)
