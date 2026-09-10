import json
from pathlib import Path


def test_reference_results_have_distinct_physical_classifications():
    results = json.loads((Path(__file__).parents[1] / "results.json").read_text(encoding="utf-8"))
    classification = results["classification"]

    assert "BIANCHI_VALID" in classification
    assert "ENERGY_MOMENTUM_CONSERVATION_VALID" in classification
    assert "DELTA_CONSERVATION_VALID" in classification
    assert "CONSERVATION_VALID" not in classification
    assert "CAUSAL_LOOP_VALID" in classification
    assert "FIXED_POINT_VALID" in classification
    assert "STATE_RECONSTRUCTION_VALID" in classification
    assert results["gr"]["baseline_source"] == "computed Einstein tensor of Minkowski geometry"
