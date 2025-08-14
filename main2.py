import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig, ModelSettings
from agents.run import AgentRunner, set_default_agent_runner
from dotenv import load_dotenv
import asyncio

load_dotenv()
gemini_api = os.getenv('GEMINI_API_KEY')
if not gemini_api:
    raise ValueError("api key not set")

external_client = AsyncOpenAI(
    api_key=gemini_api,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.5-flash"
)

config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)



class CustomAgentRunner(AgentRunner):
    async def run(self, starting_agent, input, **kwargs):
        return await super().run(starting_agent, input, **kwargs)
    
set_default_agent_runner(CustomAgentRunner())

# More focused vocabulary
focused_agent = Agent(
    name="Focused",
    instructions="you are a focus agent, you give answers to query in focused and consistent way",
    model=model,
    model_settings=ModelSettings(
        top_p=0.3,              # Use only top 30% of vocabulary
        #frequency_penalty=0.5,   # Avoid repeating words
        presence_penalty=0.3     # Encourage new topics
    )
)

agent= Agent(
    name="learning agent",
    instructions="you are a helpful assistant that help in learning",
    model=model,
    model_settings=ModelSettings(
        temperature=0.8,
        max_tokens=1000
    )
)


async def main():
    runner = CustomAgentRunner()
    result = await runner.run(focused_agent, "write about wisdom", run_config=config)
    print(result.final_output)

if __name__=='__main__':
    asyncio.run(main())