import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, trace
from agents import RawResponsesStreamEvent
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set")

# Define the OpenAI client
openai_client = AsyncOpenAI(api_key=openai_api_key)

# Define the model
model = OpenAIChatCompletionsModel(
    openai_client=openai_client,
    model="gpt-4o-mini"  # change to the model you want
)

# Create the agent
assistant_agent = Agent(
    name="assistant_agent",
    instructions="You are a helpful assistant.",
    model=model
)

# Enable tracing in the config
config = RunConfig(
    model=model,
    model_provider=openai_client,
    tracing_disabled=False
)

async def main():
    inputs = [{"content": "Hello, can you tell me a fun fact?", "role": "user"}]

    with trace("Simple assistant trace"):
        result = Runner.run_streamed(
            assistant_agent,
            input=inputs,
            run_config=config
        )
        async for event in result.stream_events():
            if not isinstance(event, RawResponsesStreamEvent):
                continue
            data = event.data
            if isinstance(data, ResponseTextDeltaEvent):
                print(data.delta, end="", flush=True)
            elif isinstance(data, ResponseContentPartDoneEvent):
                print("\n")

if __name__ == "__main__":
    asyncio.run(main())
