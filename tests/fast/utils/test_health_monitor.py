from miles.utils.health_monitor import compute_kill_list


class TestComputeKillListBasic:
    def test_empty_failures(self):
        assert compute_kill_list([], total_engines=8, engines_per_node=1, max_kill_ratio=0.5) == []

    def test_single_failure(self):
        result = compute_kill_list([3], total_engines=8, engines_per_node=1, max_kill_ratio=0.5)
        assert result == [3]

    def test_multiple_failures(self):
        result = compute_kill_list([1, 5], total_engines=8, engines_per_node=1, max_kill_ratio=0.5)
        assert result == [1, 5]

    def test_result_is_sorted(self):
        result = compute_kill_list([7, 2, 5], total_engines=8, engines_per_node=1, max_kill_ratio=1.0)
        assert result == [2, 5, 7]


class TestAntiCascade:
    def test_kills_capped_at_max_ratio(self):
        """With max_kill_ratio=0.5 and 8 engines, at most 4 can be killed per round."""
        failed = [0, 1, 2, 3, 4, 5]
        result = compute_kill_list(failed, total_engines=8, engines_per_node=1, max_kill_ratio=0.5)
        assert len(result) == 4
        assert result == [0, 1, 2, 3]

    def test_at_least_one_kill_even_with_small_ratio(self):
        result = compute_kill_list([0], total_engines=8, engines_per_node=1, max_kill_ratio=0.01)
        assert result == [0]

    def test_all_killed_when_ratio_is_one(self):
        failed = list(range(8))
        result = compute_kill_list(failed, total_engines=8, engines_per_node=1, max_kill_ratio=1.0)
        assert result == list(range(8))

    def test_small_cluster_two_engines(self):
        result = compute_kill_list([0, 1], total_engines=2, engines_per_node=1, max_kill_ratio=0.5)
        assert len(result) == 1


class TestNodeLevelExpansion:
    def test_majority_failure_expands_to_full_node(self):
        """4 engines per node. If 3/4 on node 0 fail, all 4 should be killed."""
        result = compute_kill_list([0, 1, 2], total_engines=8, engines_per_node=4, max_kill_ratio=1.0)
        assert result == [0, 1, 2, 3]

    def test_minority_failure_no_expansion(self):
        """4 engines per node. If only 1/4 fails, no expansion."""
        result = compute_kill_list([1], total_engines=8, engines_per_node=4, max_kill_ratio=1.0)
        assert result == [1]

    def test_exactly_half_does_not_expand(self):
        """4 engines per node. 2/4 fails -> 2 > 4//2=2 is False, no expansion."""
        result = compute_kill_list([0, 1], total_engines=8, engines_per_node=4, max_kill_ratio=1.0)
        assert result == [0, 1]

    def test_multi_node_expansion(self):
        """2 engines per node. Failures on node 0 and node 2 both expand."""
        failed = [0, 1, 4, 5]
        result = compute_kill_list(failed, total_engines=8, engines_per_node=2, max_kill_ratio=1.0)
        assert result == [0, 1, 4, 5]

    def test_expansion_only_on_affected_node(self):
        """2 engines per node. Both engines on node 1 fail -> expand; node 0 untouched."""
        result = compute_kill_list([2, 3], total_engines=8, engines_per_node=2, max_kill_ratio=1.0)
        assert result == [2, 3]

    def test_single_engine_per_node_no_expansion(self):
        """engines_per_node=1 means no grouping to expand."""
        result = compute_kill_list([0, 2], total_engines=8, engines_per_node=1, max_kill_ratio=1.0)
        assert result == [0, 2]

    def test_node_expansion_capped_at_total(self):
        """Last node may have fewer engines than engines_per_node if total isn't a multiple."""
        result = compute_kill_list([8, 9], total_engines=10, engines_per_node=4, max_kill_ratio=1.0)
        assert all(eid < 10 for eid in result)
        assert 8 in result and 9 in result


class TestCombinedConstraints:
    def test_node_expansion_then_anti_cascade(self):
        """Node expansion increases kill set, but anti-cascade caps it."""
        failed = [0, 1, 2, 4, 5, 6]
        result = compute_kill_list(failed, total_engines=8, engines_per_node=4, max_kill_ratio=0.5)
        assert len(result) == 4
