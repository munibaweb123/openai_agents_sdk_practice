from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Literal
from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace
import os
from dotenv import load_dotenv
load_dotenv()

openai_key=os.getenv('OPENAI_API_KEY')

story_outline_generator= Agent(
    name="Story outline generator",
    instructions=("you generate a very short story outline based on the user's input," 
    "If there any feedback provided, use it to improve the outline")
)

@dataclass
class EvaluationFeedback:
    feedback:str
    score:Literal["pass","needs improvement","fail"]

evaluator=Agent[None](
    name="Evaluator",
    instructions=(
        "you evaluate a story outline and decide if its good enough"
        "If its not good enough you provide feedback of whats need to be improved"
        "never give it a pass on its first try, After 3 attempts, you can give it a pass if story outline is good enough - do not go for perfection"
    ),
    output_type=EvaluationFeedback
)

async def main():
    msg=input("What kind of story would you like to hear?")
    input_items:list[TResponseInputItem]=[{"content":msg,"role":"user"}]
    latest_outline:str|None=None

    with trace("LLM as a Judge"):
        while True:
            story_outline_result= await Runner.run(
                story_outline_generator,
                input_items,

            )
            input_items=story_outline_result.to_input_list()
            latest_outline=ItemHelpers.text_message_outputs(story_outline_result.new_items)

            evaluator_result= await Runner.run(evaluator,input_items)
            result:EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator Score: {result.score}")

            if result.score == "pass":
                print("Story outline good enough, exiting")
                break
            print("rerunning with feedback")

            input_items.append({"content":f"feedback: {result.feedback}","role":"user"})
    print(f"final story outline: {latest_outline}")

if __name__ == '__main__':
    asyncio.run(main())