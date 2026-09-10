import numpy as np
import pytest

from relational_dynamics.ground_truth import build_ground_truth_worlds
from relational_dynamics.historical_diagnostics import (
    LABEL_CONFUND,
    MULTIPLE_CONFUNDS,
    NO_CONFUND_FOUND,
    ORDER_CONFUND,
    PROCESS_STRUCTURE_CONFUND,
    SPLIT_DESIGN_CONFUND,
    analyze_non_geometric_predictability,
    compare_observables_under_relabeling,
    coupling_graph_edges,
    preserves_adjacency,
    relabeled_coupling_graph_edges,
    run_non_geometric_controls,
    run_root_cause_analysis,
)

WORLDS = build_ground_truth_worlds()
SMALL_SEEDS = tuple(range(3))
PERMUTATION = (2, 0, 1, 3)


# ---------------------------------------------------------------------------
# Requirement 7: regression guard for relabeling invariance of the observable
# representation itself. This is EXPECTED TO FAIL today: the fixed-order
# construction couples physical event pairs by raw index adjacency, so a
# relabeling changes WHICH pairs are coupled at all (see AUDIT.md,
# PROCESS_STRUCTURE_CONFUND). `strict=True` means this stops being an xfail
# and starts failing the suite the moment the confound is actually fixed,
# which is the intended tripwire behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="PROCESS_STRUCTURE_CONFUND: coupling graph is index-adjacency-defined, not geometry-defined; see docs/AUDIT.md")
def test_pure_relabeling_does_not_change_the_information_content_of_observables():
    report = compare_observables_under_relabeling(WORLDS[1], PERMUTATION)
    assert not report.pairwise_information_changed
    assert not report.causal_estimate_changed
    assert not report.multipartite_information_changed
    assert not report.intervention_tensor_changed


# ---------------------------------------------------------------------------
# Coupling-graph adjacency analysis (requirements 1-3)
# ---------------------------------------------------------------------------


def test_coupling_graph_edges_are_the_fixed_index_adjacency():
    assert coupling_graph_edges(4) == [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})]


def test_identity_and_reversal_are_the_only_adjacency_preserving_permutations_for_n4():
    preserving = [permutation for permutation in _all_permutations(4) if preserves_adjacency(permutation)]
    assert set(preserving) == {(0, 1, 2, 3), (3, 2, 1, 0)}


def _all_permutations(n):
    import itertools

    return list(itertools.permutations(range(n)))


def test_relabeled_coupling_graph_edges_differ_from_original_for_a_generic_permutation():
    assert set(relabeled_coupling_graph_edges(PERMUTATION)) != set(coupling_graph_edges(4))


def test_observable_invariance_report_shows_angles_agree_on_shared_edges():
    """The coupling LAW itself (theta = f(interval)) is untouched: for the one
    physical edge present in both processes, the angle matches exactly."""
    report = compare_observables_under_relabeling(WORLDS[1], PERMUTATION)
    assert report.shared_edges == (frozenset({0, 1}),)
    assert report.angles_on_shared_edges_match is True


def test_observable_invariance_report_shows_coupling_graph_is_not_preserved():
    report = compare_observables_under_relabeling(WORLDS[1], PERMUTATION)
    assert report.coupling_graph_preserved is False
    assert report.pairwise_information_changed is True


# ---------------------------------------------------------------------------
# Predictability from non-geometric features (requirements 4/6)
# ---------------------------------------------------------------------------


def test_permutation_loss_analysis_enumerates_every_permutation():
    report = analyze_non_geometric_predictability(WORLDS[1], SMALL_SEEDS)
    assert report.permutation_count == 24
    assert np.isfinite(report.adjacency_preserving_loss_mean)
    assert np.isfinite(report.adjacency_breaking_loss_mean)


def test_permutation_enumeration_rejects_worlds_too_large_to_exhaust():
    from relational_dynamics.historical_scaling import build_scaled_worlds

    with pytest.raises(ValueError):
        analyze_non_geometric_predictability(build_scaled_worlds(6)[0], SMALL_SEEDS)


# ---------------------------------------------------------------------------
# Explicit non-geometric controls A-D (requirement 5)
# ---------------------------------------------------------------------------


def test_non_geometric_controls_cover_a_through_d():
    report = run_non_geometric_controls(WORLDS[1], SMALL_SEEDS)
    for value in (
        report.coupled_test_loss_mean,
        report.control_a_randomized_labels_test_loss_mean,
        report.control_b_randomized_strengths_test_loss_mean,
        report.control_c_randomized_order_test_loss_mean,
        report.control_d_randomized_indexing_test_loss_mean,
    ):
        assert value >= 0.0


# ---------------------------------------------------------------------------
# Root-cause classification vocabulary (requirement 10)
# ---------------------------------------------------------------------------


def test_root_cause_classification_uses_only_the_required_vocabulary():
    report = run_root_cause_analysis(WORLDS[1], SMALL_SEEDS, permutation=PERMUTATION)
    assert report.classification in {
        LABEL_CONFUND,
        ORDER_CONFUND,
        PROCESS_STRUCTURE_CONFUND,
        SPLIT_DESIGN_CONFUND,
        MULTIPLE_CONFUNDS,
        NO_CONFUND_FOUND,
    }


def test_root_cause_analysis_flags_process_structure_and_label_confunds_on_world_2():
    report = run_root_cause_analysis(WORLDS[1], SMALL_SEEDS, permutation=PERMUTATION)
    assert report.process_structure_confund is True
    assert report.label_confund is True
    assert report.classification == MULTIPLE_CONFUNDS


def test_root_cause_analysis_does_not_modify_q_or_delta():
    report = run_root_cause_analysis(WORLDS[1], SMALL_SEEDS, permutation=PERMUTATION)
    assert report.evidence["invariance_report"] is not None  # ran without BaselineGate raising
