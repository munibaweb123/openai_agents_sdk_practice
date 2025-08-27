import os
import asyncio
from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
    input_guardrail,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    output_guardrail,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    TResponseInputItem,
)
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# External client (Gemini with OpenAI-compatible API)
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model config
model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=False,
)

# ----------------------------
# Pydantic output models
# ----------------------------
class MessageOutput(BaseModel):
    response: str

class math_output(BaseModel):
    is_math_homework: bool
    reasoning: str

class pak_output(BaseModel):
    is_relevant: bool
    reasoning: str

# ----------------------------
# Guardrail helper agents
# ----------------------------
police = Agent(
    name="police",
    instructions="check if the user is asking for math homework",
    output_type=math_output,
    model="gpt-4o-mini",
)

guard = Agent(
    name="guard",
    instructions="check if the user is asking for Pakistan related query",
    output_type=pak_output,
    model="gpt-4o-mini",
)

# ----------------------------
# Guardrail functions
# ----------------------------
@output_guardrail
async def pak_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    """Check if the agent's output is relevant to Pakistan queries."""
    result = await Runner.run(guard, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_relevant is False,  # trip if NOT Pakistan-related
    )

@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Block math homework queries before they reach the main agent."""
    result = await Runner.run(police, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_homework,
    )

# ----------------------------
# Main Agents
# ----------------------------
customer_agent = Agent(
    name="Customer Support Agent",
    instructions="you are a customer support agent, you help customers with their queries",
    model="gpt-4o-mini",
    input_guardrails=[math_guardrail],
)

pakistan_agent = Agent(
    name="Pakistan Agent",
    instructions="you are a Pakistan agent, you answer Pakistan related queries",
    model="gpt-4o-mini",
    output_type=MessageOutput,
    output_guardrails=[pak_guardrail],
)

# ----------------------------
# Runner test
# ----------------------------
async def main():
    try:
        await Runner.run(
            pakistan_agent,
            "Hello, what is the prime minister of India?",  # ❌ not Pakistan-related
            run_config=config,
        )
        print("Guardrail didn't trip - this is unexpected")

    except InputGuardrailTripwireTriggered:
        print("❌ Math homework guardrail tripped")
    except OutputGuardrailTripwireTriggered:
        print("❌ Query is not relevant to Pakistan (output guardrail tripped)")

asyncio.run(main())
