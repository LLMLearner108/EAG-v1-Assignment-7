# basic import
from mcp.server.fastmcp import FastMCP
import sys
from mcp_schemas import *
from utils import *
import time

# instantiate an MCP server client
mcp = FastMCP("RAG Agent")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Sum of the two numbers
    """
    return a + b


@mcp.tool()
def show_reasoning(input: ShowReasoningInput) -> TextContentOutput:
    """
    Displays the reasoning for solving a particular problem to the user.

    This tool:
    1. Takes structured reasoning input
    2. Formats it for display
    3. Returns a text output with the reasoning

    Args:
        input (ShowReasoningInput): Input parameters containing:
            - reasoning: A list of ReasoningStep objects containing:
                - step_text (str): The reasoning text
                - type_of_reasoning list(str): Type of reasoning for eg. (batting, bowling, fielding, captaincy, pitch_conditions, weather_conditions, toss_decision, powerplay_strategy, death_overs_strategy, player_selection, auction_strategy, foreign_player_usage, impact_player_usage, team_combination, injury_management, crowd_influence, momentum_shift, commentary, brand_visibility, social_media_effect, fan_engagement, technology_usage)
                - reasoning_issues (str): Issues in the reasoning
                - are_there_any_issues_in_reasoning (bool): Boolean indicating if there are issues

    Returns:
        TextContentOutput: A message showing the reasoning for solving the problem
    """
    return TextContentOutput(
        result=[
            {
                "type": "text",
                "text": f"Reasoning:\n'{input.reasoning}' successfully determined and displayed to the user",
            }
        ]
    )


@mcp.tool()
def verify_step(input: VerifyStepInput) -> TextContentOutput:
    """
    Verifies if the last performed step is correct or not.

    This tool:
    1. Takes verification input
    2. Validates the last step
    3. Returns a verification message

    Args:
        input (VerifyStepInput): Input parameters containing:
            - verification (str): Justification or verification of the last performed step

    Returns:
        TextContentOutput: A message showing the verification of the last performed step
    """
    return TextContentOutput(
        result=[
            {
                "type": "text",
                "text": f"Verification:\n'{input.verification}' done for the last performed step",
            }
        ]
    )


@mcp.tool()
def search_documents(input: SearchDocumentsInput) -> TextContentOutput:
    """
    Search for relevant content from uploaded documents.

    This tool:
    1. Takes a search query
    2. Searches the FAISS index for relevant chunks
    3. Returns matching document chunks with metadata

    Args:
        input (SearchDocumentsInput): Input parameters containing:
            - query (str): The query which was asked by the user which I need to answer

    Returns:
        TextContentOutput: A message showing the chunks or documents fetched from the database

    Raises:
        Exception: If there's an error searching the documents
    """
    print("Entered search documents")
    ensure_faiss_ready()

    query = input.query
    print("SEARCH", f"Query: {query}")
    try:
        index = faiss.read_index(str(INDEX_CACHE / "index.bin"))
        metadata = json.loads((INDEX_CACHE / "metadata.json").read_text())
        query_vec = get_embedding(query, OLLAMA_URL, EMBEDING_MODEL_NAME).reshape(1, -1)
        D, I = index.search(query_vec, k=5)
        results = []
        for idx in I[0]:
            data = metadata[idx]
            results.append(
                {"chunk": data["chunk"], "source": data["doc"], "id": data["chunk_id"]}
            )
        return TextContentOutput(result=results)
    except Exception as e:
        return TextContentOutput(
            result=[
                {
                    "error_message_from_search_documents_tool": f"ERROR: Failed to search: {str(e)}"
                }
            ]
        )


@mcp.tool()
def generate_answer(input: GenerateAnswerInput) -> TextContentOutput:
    """
    Given a user query, generate the answer for the query.

    This tool:
    1. Takes structured input about the answer generation process
    2. Validates the input and context
    3. Returns the generated answer with attribution

    Args:
        input (GenerateAnswerInput): Input parameters containing:
            - do_i_have_enough_context_to_generate_answer (bool): True if I have relevant information with which I can generate an answer without hallucinating
            - initial_short_draft_answer(str): A small draft answer before the actual answer is generated
            - critique_of_initial_answer(str): Critiquing the generated draft answer, criticism should have information about whether the answer adheres to the context, does it partially or fully answer the given question, is it impossible to answer this question with the knowledge that I currently have etc.
            - generated_answer(str): The answer to the user's query
            - chunks_used_to_generate_answer: List[ChunkAttribution]: A list of ChunkAttribution objects. ChunkAttributionObjects contain:
                * chunk_text(str): The part of the text from the chunk which was used to answer the given query
                * chunk_id(str): The ID of the chunk from which the above part was extracted
                * chunk_source(str): The source document from which this chunk was extracted
            - have_i_used_any_external_knowledge_to_answer(bool): Based on the chunks obtained above and the answer generated, whether any knowledge apart from what is present in the chunks or what I have been told was used to generate the answer

    Returns:
        TextContentOutput: A message showing generated answer with attribution
    """
    if (
        input.do_i_have_enough_context_to_generate_answer
        and input.generated_answer
        and not input.have_i_used_any_external_knowledge_to_answer
    ):
        return TextContentOutput(
            result=[
                {"answer": input.generated_answer},
                {
                    "chunks used to fetch the answer": input.chunks_used_to_generate_answer
                },
            ]
        )
    else:
        return TextContentOutput(
            result=[{"message": "Answer could not be found from the given documents"}]
        )


if __name__ == "__main__":
    print("STARTING THE SERVER AT AMAZING LOCATION")

    ensure_faiss_ready()

    mcp.run(transport="stdio")
