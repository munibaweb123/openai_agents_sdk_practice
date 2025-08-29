import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from agents import Agent, RunConfig, RunContextWrapper, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.5-flash",
)

config = RunConfig(
    model=llm_model,
    model_provider=external_client,
    tracing_disabled=False,
)


@dataclass
class UserInfo:
    name:str
    uid:int

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo])->str:
    return f"User {wrapper.context.name} is 30 years old."

async def main():
    user_1 = UserInfo(name="Alice",uid=101)

    # define agent that use tool define above
    agent = Agent(
        name="user_info_agent",
        instructions="You are a helpful assistant, use tool to fetch user age.",
        tools=[fetch_user_age],
        #model=llm_model,
    )

    result= await Runner.run(
        starting_agent=agent,
        input="what is the age of the user?",
        context=user_1,
        run_config=config,
    )

    print(result.final_output)

asyncio.run(main())