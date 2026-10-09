"""Execute the structured-observation examples published in the documentation."""

from pathlib import Path
import runpy

import pytest


SNIPPET_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "snippets" / "observation_results"
)
SNIPPETS = tuple(sorted(SNIPPET_DIR.glob("*.py")))


def test_structured_observation_snippet_set_is_explicit() -> None:
    assert tuple(path.name for path in SNIPPETS) == (
        "legacy_response_only.py",
        "legacy_rt_choice.py",
        "response_only_omission.py",
        "rt_confidence.py",
    )


@pytest.mark.parametrize("snippet", SNIPPETS, ids=lambda path: path.stem)
def test_structured_observation_snippet_executes(snippet: Path) -> None:
    namespace = runpy.run_path(str(snippet))

    assert namespace["validated_result"]["observations"].ndim == 3
