from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ECQLGeneration(BaseModel):
    reasoning: str = Field(default="", description="One-sentence rationale.")
    ecql_string: str = Field(description="The final valid ECQL filter")


class SpatialTargetDef(BaseModel):
    id: str = Field(description="Stable target key such as g1 or r1.")

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("SpatialTargetDef.id must not be blank.")
        return v

    kind: Literal["explicit_geometry", "spatial_reference"] = Field(
        description="Whether the target is explicit WKT geometry or textual spatial reference."
    )
    value: str = Field(description="WKT value for explicit geometry or location text for references.")
    role: Literal["primary_area", "secondary_area", "proximity_anchor", "unspecified"] = Field(
        default="unspecified",
        description="Optional semantic role of this target in the user request.",
    )
    crs: str | None = Field(
        default=None,
        description="Optional CRS for explicit geometry values (defaults to EPSG:4326 downstream).",
    )
    required: bool = Field(
        default=True,
        description="If false, unresolved target should not fail the full request.",
    )


class SpatialPredicateBindingDef(BaseModel):
    id: str = Field(description="Stable predicate key such as p1 or p2.")

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("SpatialPredicateBindingDef.id must not be blank.")
        return v

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
    ] = Field(description="The exact OGC spatial predicate to apply.")
    target_ids: list[str] = Field(
        description="Referenced spatial target ids. Unary predicates use one id, binary use two ids."
    )
    distance: float | None = Field(
        default=None,
        description="Distance value for DWITHIN/BEYOND predicates.",
    )
    units: Literal["meters", "kilometers", "feet", "statute miles", "nautical miles"] | None = Field(
        default=None,
        description="Distance units for DWITHIN/BEYOND predicates.",
    )
    join_with_next: Literal["AND", "OR"] = Field(
        default="AND",
        description="Boolean join operator to the next predicate binding.",
    )
    required: bool = Field(
        default=True,
        description="If false, unresolved predicate binding should not fail the full request.",
    )

    @field_validator("target_ids")
    @classmethod
    def target_ids_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_ids must contain at least one target id.")
        return v

    @model_validator(mode="after")
    def dwithin_beyond_requires_distance_and_units(self) -> "SpatialPredicateBindingDef":
        if self.predicate in ("DWITHIN", "BEYOND"):
            if self.distance is None:
                raise ValueError(f"{self.predicate} requires 'distance'.")
            if self.distance <= 0:
                raise ValueError(f"{self.predicate} 'distance' must be positive.")
            if self.units is None:
                raise ValueError(f"{self.predicate} requires 'units'.")
        return self


class AnalyzedIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    layer_subject: str | None = Field(
        default=None,
        description="The core entity being searched for (e.g., 'hospitals', 'roads').",
    )
    attribute_hints: list[str] | None = Field(
        default=None,
        description="Conditions/filters applied to subject (e.g., ['capacity > 100']).",
    )
    spatial_targets: list[SpatialTargetDef] | None = Field(
        default_factory=list,
        description="Id-bound spatial targets for deterministic multi-geometry/reference requests.",
    )
    spatial_predicates: list[SpatialPredicateBindingDef] | None = Field(
        default_factory=list,
        description="Structured id-bound spatial predicate bindings referencing spatial_targets by id.",
    )
