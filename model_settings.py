from agents import Agent, ModelSettings, Runner
import os
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    raise ValueError('openai api key not set')

focused_agent = Agent(
    name="Focused",
    instructions="you are a focus agent, you give answers to query in focused and consistent way",
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        top_p=0.3,              # Use only top 30% of vocabulary
        frequency_penalty=0.5,   # Avoid repeating words
        presence_penalty=0.3     # Encourage new topics
    )
)

result=Runner.run_sync(focused_agent,"what is openai agents sdk?")
print(result.final_output)

