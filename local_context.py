import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from agents import Agent, ModelSettings, Runner,OpenAIChatCompletionsModel, function_tool, RunContextWrapper
from openai import AsyncOpenAI 


_:bool = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key and not openai_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
llm_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.5-flash",
)

@dataclass
class UserContext:
    username:str
    email:str

@function_tool
async def search(local_context:RunContextWrapper[UserContext], query:str)->str:
    
    await asyncio.sleep(2)
    print(f"Searching for '{query}' for user {local_context.context.username} with email {local_context.context.email}")
    #return "no result found"

async def special_prompt(special_context:RunContextWrapper[UserContext], agent:Agent[UserContext])->str:
    # who is user?
    # which agent?
    print(f"\n User:{special_context.context}, \n Agent: {agent.name}\n")
    return f"you are math expert. User {special_context.context.username}, Agent: {agent.name}. Please answer math related queries. "

math_agent = Agent(
    name="genius",
    instructions=special_prompt,
    tools=[search],
    model_settings=ModelSettings(tool_choice='required'),
    model=llm_model,
)

async def call_agent():
    user_context = UserContext(username="Alice", email="alice@example.com")

    output = await Runner.run(
        starting_agent=math_agent,
        input="what is 12 multiplied by 15?",
        context=user_context,
    )
    print(f"\nOutput:{output.final_output}")

asyncio.run(call_agent())