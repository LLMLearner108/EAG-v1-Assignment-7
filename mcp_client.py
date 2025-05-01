import os, time
import logging
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from openai import OpenAI
from concurrent.futures import TimeoutError
from utils import *
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from sub_prompts import *
from uuid import uuid4
from memory import *
from perception import *
from decision import *
from action import *

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("mcp_client")

max_iterations = 10
last_response = None


async def main(query):
    """
    Main function to execute the RAG agent for answering IPL-related questions.

    This function:
    1. Initializes the memory and decision components
    2. Creates an MCP session for tool execution
    3. Processes the user's query through the perception phase
    4. Executes the decision-action loop until completion
    5. Handles errors and cleanup

    Args:
        query (str): The user's question about IPL

    The function follows these steps:
    1. Initialize memory and decision components
    2. Create MCP session and get available tools
    3. Process user query through perception
    4. Execute decision-action loop:
       - Retrieve relevant memories
       - Make decisions about next actions
       - Execute chosen actions
       - Continue until completion or max iterations
    5. Handle errors and cleanup resources
    """
    print("Starting main execution...")

    # Create the memory object to store the user preferences and the iteration responses
    mem = Memory(logger)

    # Create a decision object to make decisions for our use case
    decision = Decision(client, logger)

    logger.info(f"User Query:\n{query}\n")

    current_iteration = 0

    session_id = f"session-{int(time.time())}"

    try:
        # Create a single MCP server connection
        print("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_action_server.py"],
            cwd="/Users/vinayaknayak/Desktop/EAGv1/EAG-v1-Assignment-7/",
        )

        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()

                # Get available tools
                print("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                tools_description = get_description_from_tools(tools)

                # Create an action object to execute the tools for our taks
                action = Action(tools, mem, logger)

                # Create perception output
                perception_prompt = build_perception_prompt(tools_description, query)
                perception_response_text = await generate_with_timeout(
                    client, perception_prompt
                )

                # Validate the perceived output
                perception_object = PerceptionObject.model_validate_json(
                    perception_response_text
                )

                logger.info(
                    f"\nPerceived the user's task and extracted this information about the task:\n{perception_response_text}"
                )

                while current_iteration < max_iterations:

                    # Introduce a sleep to be generous to the cloud provider
                    # For free tier, we have a max of 15 requests per minute
                    await asyncio.sleep(2)

                    logger.info(f"\n--- Iteration {current_iteration + 1} ---")

                    retrieved_memories = mem.retrieve(
                        query=query, top_k=3, session_filter=session_id
                    )

                    try:

                        function_call = await decision.decide(
                            query, perception_object, tools, retrieved_memories
                        )

                        to_continue = await action.execute_action(
                            query,
                            function_call,
                            session,
                            session_id,
                            current_iteration + 1,
                        )

                        if not to_continue:
                            break

                    except Exception as e:
                        logger.error(f"Error details: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        break

                    current_iteration += 1

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        pass


if __name__ == "__main__":
    query = input("Enter your query please.\n\n")
    asyncio.run(main(query))
