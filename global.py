from agents import Agent, set_default_openai_api,set_default_openai_client,Runner, set_tracing_disabled
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

set_tracing_disabled(True)
set_default_openai_api('chat_completions')

gemini_api = os.getenv('GEMINI_API_KEY')

external_client = AsyncOpenAI(
    api_key=gemini_api,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(external_client)

agent: Agent = Agent(
    name="Default Agent",
    instructions="You are a helpful assistant",
    model="gemini-2.0-flash",
)



result = Runner.run_sync(agent, "Write a poem about AI in haiku style", )
print(result.final_output)