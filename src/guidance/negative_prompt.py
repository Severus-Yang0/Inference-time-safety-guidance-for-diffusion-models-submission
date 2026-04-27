"""Method M2: Negative prompt baseline.

This is *not* a guidance hook — it's just a different argument to `sample()`.
The `make()` helper returns the prompt string to pass as `negative_prompt`.
Centralized here so the runner can treat all methods uniformly.
"""
DEFAULT_NEGATIVE = (
    "violence, weapon, blood, gore, nudity, sexual content, hate symbol, "
    "offensive, disturbing, graphic, explicit"
)


def negative_prompt_text() -> str:
    return DEFAULT_NEGATIVE
