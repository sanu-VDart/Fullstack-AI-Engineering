from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import List

class IrisRequest(BaseModel):
    features: Annotated[
        List[float],
        Field(
            min_length=4,
            max_length=4,
            json_schema_extra={
                "example": [5.1, 3.5, 1.4, 0.2]
            }
        )
    ]


