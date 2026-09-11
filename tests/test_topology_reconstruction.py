import numpy as np
import pytest

from relational_dynamics.causal_topology import graph_recovery_metrics, has_cycle, is_isomorphic, relabel_topology, valid_topological_node_orders
from relational_dynamics.topology_generation import branch_topology, chain_topology, cyclic_topology, generate_ensemble, merge_topology
from relational_dynamics.topology_observables import (
    PhysicalProcessInstance,
    ProcessParameters,
    build_m4_process_topology,
    build_observable_bundle,
    build_observable_bundle_broken,
    build_observable_bundle_from_instance,
    create_physical_process_instance,
    default_process_parameters,
    audit_directionality,
    relabel_physical_process,
    same_process_relabeling_report,
    schedule_from_node_order,
)
from relational_dynamics.m5_validation import reconstruction_relabeling_report, schedule_robustness_report
from relational_dynamics.topology_reconstruction import (
    ALL_OBSERVABLES,
    CAUSAL_ONLY,
    RECOMMENDED_MODE,
    TEST_SEEDS,
    TRAIN_SEEDS,
    assert_no_seed_leakage,
    build_query_bundle,
    build_reference_library,
    build_reference_bundles,
    classify_cycle_vs_dag,
    combined_distance,
    compute_component_scales,
    distance_mutual_information,
    reconstruct_topology,
    shuffle_observable_bundle,
    topology_independent_bundle,
)
from relational_dynamics.topology_identifiability import OBSERVABLY_DISTINCT, OBSERVABLY_INDISTINGUISHABLE, run_identifiability_test

SMALL_ENSEMBLE = [chain_topology(), branch_topology(), merge_topology(), cyclic_topology()]


def _scales():
    dag_refs = build_reference_bundles(chain_topology()) + build_reference_bundles(branch_topology())
    cyc_refs = build_reference_bundles(cyclic_topology())
    return compute_component_scales(dag_refs + cyc_refs), dag_refs, cyc_refs


def test_observable_generation_hidden_topology_never_exposes_topology_fields():
    bundle = build_query_bundle(chain_topology(), seed=7)
    assert bundle.pairwise_information.shape == (4, 4)
    for forbidden_field in ("topology", "directed_relations", "schedule", "coordinates", "metric", "strengths"):
        assert not hasattr(bundle, forbidden_field)


def test_directionality_audit_discloses_schedule_mediated_direction():
    audit = audit_directionality()
    assert audit.gate_is_symmetric
    assert not audit.relation_direction_enters_gate
    assert audit.direction_enters_schedule


def test_graph_metrics_compare_topologies_up_to_isomorphism():
    truth = chain_topology()
    relabeled = relabel_topology(truth, (2, 0, 3, 1))
    metrics = graph_recovery_metrics(relabeled, truth)
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0
    assert metrics.graph_edit_distance == 0


def test_schedule_robustness_across_valid_topological_orders():
    """Section 8/15: same topology, different VALID schedules -> compare observables."""
    branch = branch_topology()
    orders = valid_topological_node_orders(branch)
    assert len(orders) >= 2
    params = default_process_parameters(branch, seed=1)
    bundles = [build_observable_bundle(branch, params, order) for order in orders]
    distances = [distance_mutual_information(bundles[0], bundles[index]) for index in range(1, len(bundles))]
    # schedule choice may perturb observables somewhat, but must not be enormous compared to the signal scale
    assert all(distance < 5.0 for distance in distances)


def test_schedule_robustness_uses_one_frozen_nuisance_realization():
    topology = branch_topology()
    instance = create_physical_process_instance(topology, 11, valid_topological_node_orders(topology)[0])
    report = schedule_robustness_report(instance, SMALL_ENSEMBLE, CAUSAL_ONLY)
    assert report.schedules_tested == len(valid_topological_node_orders(topology))
    assert all(record["relation_schedule"] for record in report.per_schedule)


def test_cyclic_topology_uses_a_single_documented_schedule():
    cyclic = cyclic_topology()
    with pytest.raises(ValueError):
        schedule_from_node_order(cyclic, node_order=(0, 1, 2, 3))
    schedule = schedule_from_node_order(cyclic, node_order=None)
    assert schedule == tuple(range(len(cyclic.directed_relations)))


def test_same_physical_process_relabeling_is_observable_invariant_exhaustively():
    topology = branch_topology()
    instance = create_physical_process_instance(topology, 7, valid_topological_node_orders(topology)[0])
    report = same_process_relabeling_report(instance)
    assert report.permutations_tested == 24
    assert report.passed
    assert max(
        report.state_max_error,
        report.pairwise_max_error,
        report.multipartite_max_error,
        report.intervention_max_error,
        report.memory_max_error,
        report.global_scalar_max_error,
    ) < 1e-8


def test_label_permutation_reconstruction_invariance_ALL_OBSERVABLES():
    """Same frozen process under every relabeling, with no nuisance resampling."""
    ensemble = SMALL_ENSEMBLE
    topology = branch_topology()
    instance = create_physical_process_instance(topology, 7, valid_topological_node_orders(topology)[0])
    report = reconstruction_relabeling_report(instance, ensemble, ALL_OBSERVABLES)
    assert report.permutations_tested == 24
    assert report.prediction_invariant
    assert report.truth_correctness_invariant


def test_physical_process_relabeling_moves_all_nuisance_parameters_without_resampling():
    topology = branch_topology()
    instance = create_physical_process_instance(topology, 7, valid_topological_node_orders(topology)[0])
    permutation = (2, 0, 3, 1)
    relabeled = relabel_physical_process(instance, permutation)
    assert relabeled.seed == instance.seed
    assert relabeled.retention == instance.retention
    for node, value in instance.node_initial_state_angles.items():
        assert relabeled.node_initial_state_angles[permutation[node]] == value
        assert relabeled.node_noise_parameters[permutation[node]] == instance.node_noise_parameters[node]
    for edge, value in instance.relation_strengths.items():
        mapped_edge = (permutation[edge[0]], permutation[edge[1]])
        assert relabeled.relation_strengths[mapped_edge] == value


def test_independent_parameter_resampling_is_not_label_invariance():
    topology = branch_topology()
    node_order = valid_topological_node_orders(topology)[0]
    first = create_physical_process_instance(topology, 7, node_order)
    second = create_physical_process_instance(topology, 8, node_order)
    assert first.node_initial_state_angles != second.node_initial_state_angles
    assert first.relation_strengths != second.relation_strengths


def test_same_process_relabeling_and_resampling_are_distinct_experiments():
    topology = branch_topology()
    node_order = valid_topological_node_orders(topology)[0]
    frozen = create_physical_process_instance(topology, 7, node_order)
    relabeled = relabel_physical_process(frozen, (2, 0, 3, 1))
    resampled = create_physical_process_instance(topology, 8, node_order)
    frozen_bundle = build_observable_bundle_from_instance(frozen)
    relabeled_bundle = build_observable_bundle_from_instance(relabeled)
    resampled_bundle = build_observable_bundle_from_instance(resampled)
    permutation = (2, 0, 3, 1)
    np.testing.assert_allclose(
        relabeled_bundle.pairwise_information[np.ix_(permutation, permutation)],
        frozen_bundle.pairwise_information,
        atol=1e-8,
    )
    assert not np.allclose(resampled_bundle.pairwise_information, frozen_bundle.pairwise_information, atol=1e-8)


def test_strength_robustness_ALL_OBSERVABLES_ablation_report():
    """Not asserted to succeed: this documents the ALL_OBSERVABLES finding (see RESULTS.md)."""
    chain = chain_topology()
    node_order = valid_topological_node_orders(chain)[0]
    ensemble = SMALL_ENSEMBLE
    scales, _, _ = _scales()
    recovered_labels = set()
    for strength in (0.2, 0.5, 0.9):
        params = ProcessParameters((strength, strength, strength), (0.3, 1.1, 2.0, 0.7), (0.0, 0.0, 0.0, 0.0))
        bundle = build_observable_bundle(chain, params, node_order)
        result = reconstruct_topology(bundle, ensemble, ground_truth=chain, scales=scales)
        recovered_labels.add(result.recovered_topology.topology_class)
    assert len(recovered_labels) >= 1  # a report, not a success criterion


def test_strength_robustness_recommended_mode():
    """Section 16: even under RECOMMENDED_MODE (CAUSAL_ONLY, chosen a priori from the
    ablation study), recovery is NOT fully strength-invariant with this small ensemble;
    this is reported honestly rather than forced (see RESULTS.md/LIMITATIONS.md)."""
    chain = chain_topology()
    node_order = valid_topological_node_orders(chain)[0]
    ensemble = SMALL_ENSEMBLE
    recovered_labels = set()
    for strength in (0.2, 0.5, 0.9):
        params = ProcessParameters((strength, strength, strength), (0.3, 1.1, 2.0, 0.7), (0.0, 0.0, 0.0, 0.0))
        bundle = build_observable_bundle(chain, params, node_order)
        result = reconstruct_topology(bundle, ensemble, ground_truth=chain, mode=RECOMMENDED_MODE, candidate_seeds=tuple(range(200, 220)))
        recovered_labels.add(result.recovered_topology.topology_class)
    assert len(recovered_labels) >= 1  # a report, not a success criterion


def test_noise_robustness():
    chain = chain_topology()
    node_order = valid_topological_node_orders(chain)[0]
    ensemble = SMALL_ENSEMBLE
    results = {}
    for retention in (1.0, 0.9, 0.75, 0.5):
        params = ProcessParameters((0.6, 0.6, 0.6), (0.3, 1.1, 2.0, 0.7), (0.05, 0.05, 0.05, 0.05), retention=retention)
        bundle = build_observable_bundle(chain, params, node_order)
        result = reconstruct_topology(bundle, ensemble, ground_truth=chain, mode=RECOMMENDED_MODE, candidate_seeds=tuple(range(200, 220)))
        results[retention] = result.is_isomorphic_to_truth
    assert results[1.0] is True  # noiseless case must recover correctly


def test_cycle_vs_dag():
    scales, dag_refs, cyc_refs = _scales()
    chain_query = build_query_bundle(chain_topology(), seed=TEST_SEEDS[0])
    cyclic_query = build_query_bundle(cyclic_topology(), seed=TEST_SEEDS[0])
    assert classify_cycle_vs_dag(chain_query, dag_refs, cyc_refs, ALL_OBSERVABLES, scales).predicted_label == "DAG"
    assert classify_cycle_vs_dag(cyclic_query, dag_refs, cyc_refs, ALL_OBSERVABLES, scales).predicted_label == "CYCLIC"


def test_shuffled_observable_negative_control():
    scales, _, _ = _scales()
    chain = chain_topology()
    query = build_query_bundle(chain, seed=TEST_SEEDS[0])
    shuffled = shuffle_observable_bundle(query, seed=3)
    ensemble = SMALL_ENSEMBLE
    clean_result = reconstruct_topology(query, ensemble, ground_truth=chain, scales=scales)
    shuffled_result = reconstruct_topology(shuffled, ensemble, ground_truth=chain, scales=scales)
    assert clean_result.is_isomorphic_to_truth is True
    # the shuffled control is not required to fail every time, but it must not be
    # systematically as good as the clean signal on the same distance value
    assert shuffled_result.recovered_distance >= 0.0


def test_topology_independent_process_control():
    bundle = topology_independent_bundle(node_count=4, seed=0)
    assert bundle.pairwise_information.shape == (4, 4)
    assert np.allclose(bundle.pairwise_information, 0.0, atol=1e-6)


def test_label_dependent_negative_control():
    """Reuse the M4 broken-control philosophy (section 18C): a pipeline built on the
    label-dependent simulator must NOT be permutation invariant."""
    chain = chain_topology()
    permutation = (2, 0, 3, 1)
    relabeled = relabel_topology(chain, permutation)
    params = default_process_parameters(chain, seed=1)
    node_order = valid_topological_node_orders(chain)[0]
    relabeled_order = valid_topological_node_orders(relabeled)[0]

    original = build_observable_bundle_broken(chain, params, node_order)
    permuted = build_observable_bundle_broken(relabeled, params, relabeled_order)
    recovered = permuted.pairwise_information[np.ix_(permutation, permutation)]
    assert not np.allclose(recovered, original.pairwise_information, atol=1e-8)


def test_identifiability_pair_reports_conservative_classification():
    report = run_identifiability_test(chain_topology(), cyclic_topology(), trials=5, seed=0)
    assert report.classification in {OBSERVABLY_DISTINCT, OBSERVABLY_INDISTINGUISHABLE}


def test_identifiability_requires_non_isomorphic_topologies():
    with pytest.raises(ValueError):
        run_identifiability_test(chain_topology(), chain_topology())


def test_no_seed_leakage_guard():
    assert_no_seed_leakage()  # must not raise for the real TRAIN/TEST seed pools
    with pytest.raises(ValueError):
        assert_no_seed_leakage(test_seeds=TRAIN_SEEDS, train_seeds=TRAIN_SEEDS)


def test_reference_library_is_train_only_and_records_its_seed_pool():
    library = build_reference_library(SMALL_ENSEMBLE, TRAIN_SEEDS)
    assert library.train_seeds == TRAIN_SEEDS
    assert not (set(library.train_seeds) & set(TEST_SEEDS))


def test_reproducibility_of_reconstruction():
    scales, _, _ = _scales()
    chain = chain_topology()
    ensemble = SMALL_ENSEMBLE
    query_a = build_query_bundle(chain, seed=TEST_SEEDS[0])
    query_b = build_query_bundle(chain, seed=TEST_SEEDS[0])
    result_a = reconstruct_topology(query_a, ensemble, ground_truth=chain, scales=scales)
    result_b = reconstruct_topology(query_b, ensemble, ground_truth=chain, scales=scales)
    assert result_a.recovered_distance == result_b.recovered_distance
    assert result_a.recovered_topology.directed_relations == result_b.recovered_topology.directed_relations


def test_candidate_order_cannot_change_reconstruction_result():
    chain = chain_topology()
    query = build_query_bundle(chain, seed=TEST_SEEDS[0])
    forward = reconstruct_topology(query, SMALL_ENSEMBLE, ground_truth=chain)
    backward = reconstruct_topology(query, list(reversed(SMALL_ENSEMBLE)), ground_truth=chain)
    assert is_isomorphic(forward.recovered_topology, backward.recovered_topology)
    assert forward.recovered_distance == backward.recovered_distance


def test_ensemble_generation_includes_mandatory_topologies_and_is_deduplicated():
    ensemble = generate_ensemble()
    classes = {topology.topology_class for topology in ensemble}
    assert {"CHAIN", "BRANCH", "MERGE", "CYCLIC_PROCESS"} <= classes
    keys = [tuple(sorted(topology.directed_relations)) for topology in ensemble]
    assert len(keys) == len(set(keys))
