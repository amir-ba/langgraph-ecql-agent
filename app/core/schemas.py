from pydantic import BaseModel, Field
from typing import Literal


class ECQLGeneration(BaseModel):
    reasoning: str = Field(description="Why this ECQL was constructed")
    ecql_string: str = Field(description="The final valid ECQL filter")


class AnalyzedIntent(BaseModel):
    intent: Literal["spatial_query", "general_chat", "irrelevant"] = Field(
        description=(
            "Classify the user intent. 'spatial_query' if they want map/layer data. "
            "'general_chat' for greetings/conversations."
        )
    )
    general_response: str | None = Field(
        default=None,
        description=(
            "If intent is 'general_chat' or 'irrelevant', write the conversational response here. "
            "Otherwise, leave null."
        ),
    )
    spatial_reference: str | None = Field(
        default=None,
        description="The location name (e.g., 'Berlin', 'Central Park'). Null if none mentioned.",
    )
    spatial_relationship: str | None = Field(
        default=None,
        description="Spatial relationship (e.g., 'within 5km', 'intersecting'). Null if none.",
    )
    layer_subject: str | None = Field(
        default=None,
        description="The core entity being searched for (e.g., 'hospitals', 'roads').",
    )
    attribute_hints: list[str] | None = Field(
        default=None,
        description="Conditions/filters applied to subject (e.g., ['capacity > 100']).",
    )
