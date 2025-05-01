from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class ReasoningType(Enum):
    BATTING = "batting"
    BOWLING = "bowling"
    FIELDING = "fielding"
    CAPTAINCY = "captaincy"
    PITCH_CONDITIONS = "pitch_conditions"
    WEATHER_CONDITIONS = "weather_conditions"
    TOSS_DECISION = "toss_decision"
    POWERPLAY_STRATEGY = "powerplay_strategy"
    DEATH_OVERS_STRATEGY = "death_overs_strategy"
    PLAYER_SELECTION = "player_selection"
    AUCTION_STRATEGY = "auction_strategy"
    FOREIGN_PLAYER_USAGE = "foreign_player_usage"
    IMPACT_PLAYER_USAGE = "impact_player_usage"
    TEAM_COMBINATION = "team_combination"
    INJURY_MANAGEMENT = "injury_management"
    CROWD_INFLUENCE = "crowd_influence"
    MOMENTUM_SHIFT = "momentum_shift"
    COMMENTARY = "commentary"
    BRAND_VISIBILITY = "brand_visibility"
    SOCIAL_MEDIA_EFFECT = "social_media_effect"
    FAN_ENGAGEMENT = "fan_engagement"
    TECHNOLOGY_USAGE = "technology_usage"


class ReasoningStep(BaseModel):
    step_text: str
    type_of_reasoning: list[ReasoningType]
    reasoning_issues: str | None
    are_there_any_issues_in_reasoning: bool


# Input Schemas
class ShowReasoningInput(BaseModel):
    reasoning: List[ReasoningStep]


class VerifyStepInput(BaseModel):
    verification: str


class SearchDocumentsInput(BaseModel):
    query: str


class ChunkAttribution(BaseModel):
    chunk_text: str
    chunk_id: str
    chunk_source: str


class GenerateAnswerInput(BaseModel):
    do_i_have_enough_context_to_generate_answer: bool
    initial_short_draft_answer: str | None
    critique_of_initial_answer: str | None
    generated_answer: str | None
    chunks_used_to_generate_answer: List[ChunkAttribution] | None
    have_i_used_any_external_knowledge_to_answer: bool | None


# Output Schemas
class TextContentOutput(BaseModel):
    result: List[dict]


class GreetingOutput(BaseModel):
    result: str


class CodeReviewOutput(BaseModel):
    result: str


class DebugErrorOutput(BaseModel):
    result: List[dict]
