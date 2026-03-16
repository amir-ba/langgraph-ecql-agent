from pydantic import BaseModel, Field
from typing import Literal


class ECQLGeneration(BaseModel):
    reasoning: str = Field(description="Why this ECQL was constructed")
    ecql_string: str = Field(description="The final valid ECQL filter")


class SpatialFilterDef(BaseModel):
    predicate: Literal[
        "INTERSECTS",
        "WITHIN",
        "CONTAINS",
        "DISJOINT",
        "CROSSES",
        "OVERLAPS",
        "TOUCHES",
        "DWITHIN",
        "BEYOND",
    ] = Field(
        description=(
            "The exact OGC spatial predicate to apply. "
            "Use DWITHIN for explicit distance requests and WITHIN for strict containment."
        )
    )
    distance: float | None = Field(
        default=None,
        description="Distance value when predicate is DWITHIN or BEYOND.",
    )
    units: Literal["meters", "kilometers", "feet", "statute miles", "nautical miles"] | None = Field(
        default=None,
        description="Distance units when predicate is DWITHIN or BEYOND.",
    )


class AnalyzedIntent(BaseModel):
    intent: Literal["spatial_query", "irrelevant"] = Field(
        description=(
            "Classify the user intent. 'spatial_query' if they want map/layer data. "
            "'irrelevant' for anything else."
        )
    )
    general_response: str | None = Field(
        default=None,
        description=(
            "If intent is 'irrelevant', write the conversational response here. "
            "Otherwise, leave null."
        ),
    )
    spatial_reference: str | None = Field(
        default=None,
        description="The location name (e.g., 'Berlin', 'Central Park'). Null if none mentioned.",
    )
    spatial_filter: SpatialFilterDef | None = Field(
        default=None,
        description="Structured spatial relationship. Null if no location filtering is requested.",
    )
    layer_subject: str | None = Field(
        default=None,
        description="The core entity being searched for (e.g., 'hospitals', 'roads').",
    )
    attribute_hints: list[str] | None = Field(
        default=None,
        description="Conditions/filters applied to subject (e.g., ['capacity > 100']).",
    )
    explicit_coordinates: list[float] | None = Field(
        default=None,
        description="Exact point coordinates: [longitude, latitude]."
    )
    # --- NEW FIELD ---
    explicit_bbox: list[float] | None = Field(
        default=None,
        description=(
            "If the user provides an exact bounding box (4 coordinates), extract them here as a list of 4 floats: "
            "[min_longitude, min_latitude, max_longitude, max_latitude]. "
            "Example: 'bbox 13.1 52.3 13.5 52.7' becomes[13.1, 52.3, 13.5, 52.7]."
        )
    )