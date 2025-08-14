from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    RunContextWrapper,
    AgentHooks,
    Runner
)
import os
from dotenv import load_dotenv
import asyncio
from typing import Any

# Load environment variables
load_dotenv()

gemini_api = os.getenv('GEMINI_API_KEY')
if not gemini_api:
    raise ValueError("API key not set")

# Create OpenAI-style Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model configuration
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.5-flash"
)

# Optional run configuration (disable tracing)
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Custom hooks
class TestAgHooks(AgentHooks):
    def __init__(self, ag_display_name):
        self.event_counter = 0
        self.ag_display_name = ag_display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### {self.ag_display_name} {self.event_counter}: Agent {agent.name} started. Usage: {context.usage}")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(f"### {self.ag_display_name} {self.event_counter}: Agent {agent.name} ended. Usage: {context.usage}, Output: {output}")

# Create agent
start_agent = Agent(
    name="Content Moderator Agent",
    instructions="You are a content moderation agent. Watch social media content received and flag queries that need help or answer. We will answer anything about AI?",
    hooks=TestAgHooks(ag_display_name="content_moderator"),
    model=model
)

# Main execution
async def main():
    result = await Runner.run(
        start_agent,
        input="Will Agentic AI die at the end of 2025?",
        run_config=config
    )
    print(result.final_output)

asyncio.run(main())
print("--end--")
