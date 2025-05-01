from perception import PerceptionObject
from memory import MemoryItem
from sub_prompts import *
from utils import get_description_from_tools, generate_with_timeout, FunctionCall
from openai import OpenAI
from logging import Logger
from typing import List
from mcp import Tool


class Decision:
    """
    The Decision class is responsible for making decisions about which tools to execute
    based on the current state and memory. It forms the third step in the
    Perception -> Memory -> Decision -> Action framework.

    The Decision class:
    1. Constructs prompts for the LLM based on available tools and memory
    2. Makes decisions about which tool to execute next
    3. Validates the decisions before they are executed
    4. Ensures decisions align with the overall task goals

    Attributes:
        memory (Memory): Reference to the Memory instance for accessing history
        client (OpenAI): OpenAI client for making LLM calls
        logger (Logger): Logger instance for logging decision details
    """

    def __init__(
        self,
        client: OpenAI,
        logger: Logger,
    ):
        """
        Initialize the Decision class with necessary dependencies.

        Args:
            client (OpenAI): OpenAI client for LLM calls
            logger (Logger): Logger instance for logging
        """
        self.client = client
        self.logger = logger

    def _get_base_prompt(self, tools) -> str:
        """
        Construct the base prompt for decision making.

        This method:
        1. Gets tool descriptions
        2. Combines general instructions, tool descriptions, and special instructions
        3. Creates a comprehensive prompt for the LLM

        Args:
            tools (list): List of available tools

        Returns:
            str: The complete base prompt for decision making
        """
        tools_description = get_description_from_tools(tools)
        system_prompt = f"""
{general_instructions}

Here is a list of all the tools available at your disposal:
{tools_description}

{special_instructions}

{fallback_handling}

{decision_response_instruction}
"""
        return system_prompt

    async def decide(
        self,
        user_query: str,
        user_query_perception: PerceptionObject,
        tools: List[Tool],
        memory_items: List[MemoryItem],
    ) -> FunctionCall:
        """
        Make a decision about which tool to execute next.

        This method:
        1. Constructs the complete prompt including history and current query
        2. Gets the LLM's response
        3. Validates the response as a FunctionCall
        4. Returns the validated decision

        Args:
            user_query (str): The user's question or request
            user_query_perception (PerceptionObject): Structured understanding of the user's query
            tools (List[Tool]): List of available tools
            memory_items (List[MemoryItem]): Relevant memory items from previous interactions

        Returns:
            FunctionCall: The validated decision about which tool to execute next

        Raises:
            Exception: If the LLM response cannot be validated as a FunctionCall
        """
        # Construct the base prompt
        what_i_need_to_ask = self._get_base_prompt(tools)

        # Add the task that is asked for by the user
        what_i_need_to_ask += f"\nUser Query:\n{user_query}"

        # Add the perceived stuff from the user query
        what_i_need_to_ask += f"\nMy perception of user's query: {user_query_perception.model_dump_json()}"

        # Add the things which I have already done in the past based on the memory
        history = "What has happened or what I have information about\n"
        history += "\n".join(f"- {m.text}" for m in memory_items) or "None"
        what_i_need_to_ask += f"\n{history}\n"

        what_i_need_to_ask += "What should I do next?"

        try:
            response_text = await generate_with_timeout(
                self.client, what_i_need_to_ask, 60
            )
            self.logger.info(f"Decision Step Response: {response_text}")

            # Validate the model output
            function_call = FunctionCall.model_validate_json(response_text)

            self.logger.info(f"Validated the Decision Step response")
        except Exception as e:
            function_call = f"Error in parsing {response_text}\nCould not validate the function call output because {str(e)}"
            self.logger.error(f"Decision step response could not be validated")

        return function_call
