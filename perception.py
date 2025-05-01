from pydantic import BaseModel, Field
from sub_prompts import *
from typing import List, Optional, Literal


class PerceptionObject(BaseModel):
    """
    The PerceptionObject class represents the structured understanding of a task.
    It forms the first step in the Perception -> Memory -> Decision -> Action framework.

    This class:
    1. Captures the essential elements of a task
    2. Provides a structured format for task understanding
    3. Validates the perception using Pydantic models
    4. Serves as input for subsequent decision making

    """

    task: str = Field(..., description="A short description of the user's task")
    intent: str = Field(
        ...,
        description="Understand what does the user want to achieve? Be consise, terse but do not overlook anything from the user request.",
    )
    entities: List[str] | None = Field(
        ...,
        description="Entities present in the user query like person, organization, place, names, events, numbers etc.",
    )

    dialogue_act: (
        Literal[
            "question", "statement", "command", "confirmation", "greeting", "apology"
        ]
        | None
    ) = Field(
        ...,
        description="What is the nature of the statement that the user has said.",
    )


def build_perception_prompt(tools_description: str, user_query: str) -> str:
    """
    Build a comprehensive prompt for the perception phase.

    This function:
    1. Combines general instructions with tool descriptions
    2. Adds special instructions and fallback handling
    3. Includes the perception response schema
    4. Appends the user's query

    Args:
        tools_description (str): Description of available tools
        user_query (str): The user's task query

    Returns:
        str: A complete prompt for the perception phase
    """
    perception_prompt = f"""
{general_instructions}

Here is a list of all the tools available at your disposal:
{tools_description}

{special_instructions}

{fallback_handling}

{perception_response_instruction}

USER QUERY:
--------------
{user_query}
    """

    return perception_prompt
