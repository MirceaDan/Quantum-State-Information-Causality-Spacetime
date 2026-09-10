import json
from pathlib import Path

from relational_dynamics.hidden_time import audit_source_tree


def test_hidden_time_audit_distinguishes_numeric_control_from_physical_time():
    records = audit_source_tree(Path(__file__).parents[1] / "src")
    assert records
    assert all(record["allowed"] is True for record in records)
    assert not any(record["role"] == "physical_time" for record in records)
