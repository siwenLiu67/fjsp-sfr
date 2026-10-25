import json
from collections import Counter

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments.generate_random_rules import (
    generate_random_candidates,
    write_candidates,
)


def test_random_candidates_are_reproducible_balanced_and_safe():
    first = generate_random_candidates(count=120, seed=17)
    second = generate_random_candidates(count=120, seed=17)
    assert [rule.to_dict() for rule in first] == [rule.to_dict() for rule in second]
    assert Counter(int(rule.rule_type) for rule in first) == {
        1: 30,
        2: 30,
        3: 30,
        4: 30,
    }
    assert all(rule.feature_set == "theory11_plus3" for rule in first)


def test_random_candidate_file_matches_generation_schema(tmp_path):
    path = write_candidates(tmp_path / "random.json", count=12, seed=9)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidate_total"] == 12
    assert payload["parsed_total"] == 12
    assert payload["source"] == "random"
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps({"rules": [item["rule"] for item in payload["candidates"]]}),
        encoding="utf-8",
    )
    assert RuleLibrary.from_json(str(library_path)).action_size() == 12
