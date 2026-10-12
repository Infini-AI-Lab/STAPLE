# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock

import torch

from torchtitan.models.common.token_dispatcher import (
    _all_to_all_single_maybe_autograd,
    _deduplicated_dispatch_maybe_autograd,
    _DeduplicatedDispatch,
    _prepare_deduplicated_dispatch,
    AllToAllTokenDispatcher,
    DeduplicatedAllToAllTokenDispatcher,
)


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self, num_ranks: int) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(
            num_experts=1, top_k=1, score_before_experts=True
        )
        dispatcher = AllToAllTokenDispatcher(cfg)
        # Mock ep_mesh with a simple object that has .size() returning num_ranks
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = num_ranks
        dispatcher.ep_mesh = mock_mesh
        return dispatcher

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher(num_ranks)
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        _, _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
            dummy_input, tokens_per_expert_group
        )
        return permuted_indices, num_tokens_per_expert

    def test_basic_2ranks_2experts(self):
        # 2 ranks, 2 experts per rank
        # tokens_per_expert_group: [r0e0, r0e1, r1e0, r1e1] = [2, 3, 1, 4]
        tokens_per_expert_group = torch.tensor([2, 3, 1, 4])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major layout: e0r0(2), e0r1(1), e1r0(3), e1r1(4)
        # Input positions:
        #   r0e0: [0, 1], r0e1: [2, 3, 4], r1e0: [5], r1e1: [6, 7, 8, 9]
        # Output order:
        #   e0r0: [0, 1], e0r1: [5], e1r0: [2, 3, 4], e1r1: [6, 7, 8, 9]
        expected_indices = torch.tensor([0, 1, 5, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(permuted_indices, expected_indices)

        # num_tokens_per_expert: sum across ranks for each expert
        # e0: r0e0 + r1e0 = 2 + 1 = 3, e1: r0e1 + r1e1 = 3 + 4 = 7
        expected_num_tokens = torch.tensor([3, 7])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_single_rank(self):
        # 1 rank, 3 experts: no reordering needed
        tokens_per_expert_group = torch.tensor([4, 2, 5])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=1
        )

        expected_indices = torch.arange(11)
        torch.testing.assert_close(permuted_indices, expected_indices)
        torch.testing.assert_close(num_tokens_per_expert, tokens_per_expert_group)

    def test_single_expert(self):
        # 3 ranks, 1 expert per rank: no reordering needed
        tokens_per_expert_group = torch.tensor([3, 5, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=1, num_ranks=3
        )

        expected_indices = torch.arange(10)
        torch.testing.assert_close(permuted_indices, expected_indices)

        # Single expert gets all tokens
        expected_num_tokens = torch.tensor([10])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_zero_tokens_for_some_experts(self):
        # 2 ranks, 2 experts, some with zero tokens
        # [r0e0, r0e1, r1e0, r1e1] = [0, 3, 2, 0]
        tokens_per_expert_group = torch.tensor([0, 3, 2, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major: e0r0(0), e0r1(2), e1r0(3), e1r1(0)
        # Input positions: r0e0: [], r0e1: [0, 1, 2], r1e0: [3, 4], r1e1: []
        # Output order: e0r0: [], e0r1: [3, 4], e1r0: [0, 1, 2], e1r1: []
        expected_indices = torch.tensor([3, 4, 0, 1, 2])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([2, 3])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_all_zero_tokens(self):
        tokens_per_expert_group = torch.tensor([0, 0, 0, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        self.assertEqual(permuted_indices.numel(), 0)
        expected_num_tokens = torch.tensor([0, 0])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_uniform_distribution(self):
        # 3 ranks, 2 experts, uniform token counts
        # [r0e0, r0e1, r1e0, r1e1, r2e0, r2e1] = [2, 2, 2, 2, 2, 2]
        tokens_per_expert_group = torch.tensor([2, 2, 2, 2, 2, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=3
        )

        # Expert-major: e0r0(2), e0r1(2), e0r2(2), e1r0(2), e1r1(2), e1r2(2)
        # Input positions:
        #   r0e0: [0,1], r0e1: [2,3], r1e0: [4,5], r1e1: [6,7], r2e0: [8,9], r2e1: [10,11]
        # Output: e0r0[0,1], e0r1[4,5], e0r2[8,9], e1r0[2,3], e1r1[6,7], e1r2[10,11]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([6, 6])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_permutation_is_valid(self):
        # The output should be a permutation of [0, total)
        tokens_per_expert_group = torch.tensor([3, 1, 4, 1, 5, 9])
        permuted_indices, _ = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=2
        )

        total = tokens_per_expert_group.sum().item()
        self.assertEqual(permuted_indices.numel(), total)
        self.assertEqual(
            set(permuted_indices.tolist()),
            set(range(total)),
        )


class TestAllToAllDispatch(unittest.TestCase):
    def test_no_grad_dispatch_uses_non_autograd_collective(self):
        tensor = torch.arange(4)
        with (
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.all_to_all_single",
                side_effect=lambda value, *args, **kwargs: value,
            ) as functional,
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.all_to_all_single_autograd",
            ) as autograd,
            torch.no_grad(),
        ):
            actual = _all_to_all_single_maybe_autograd(
                tensor,
                [4],
                [4],
                unittest.mock.MagicMock(),
            )

        torch.testing.assert_close(actual, tensor)
        functional.assert_called_once()
        autograd.assert_not_called()

    def test_standard_dispatch_uses_local_reorder_contract(self):
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(
                num_experts=4,
                top_k=2,
                score_before_experts=False,
            )
        )
        dispatcher.ep_mesh = unittest.mock.MagicMock()
        dispatcher.ep_mesh.size.return_value = 2

        x = torch.arange(6, dtype=torch.float32).view(2, 3)
        scores = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        expert_ids = torch.tensor([[0, 2], [1, 3]])

        with (
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.torch.histc",
                return_value=torch.ones(4, dtype=torch.int64),
            ),
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.all_to_all_single",
                side_effect=lambda tensor, *args, **kwargs: tensor,
            ),
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.all_to_all_single_autograd",
                side_effect=lambda tensor, *args, **kwargs: tensor,
            ),
            unittest.mock.patch(
                "torch.ops._c10d_functional.wait_tensor",
                side_effect=lambda tensor: tensor,
            ),
        ):
            routed_input, tokens_per_expert, metadata = dispatcher.dispatch(
                x, scores, expert_ids
            )

        self.assertEqual(routed_input.shape, (4, 3))
        torch.testing.assert_close(tokens_per_expert, torch.tensor([2, 2]))
        torch.testing.assert_close(
            metadata.token_indices_experts_sorted_N,
            torch.tensor([0, 1, 0, 1]),
        )


class TestDeduplicatedDispatch(unittest.TestCase):
    def test_no_grad_dispatch_skips_custom_autograd_function(self):
        x = torch.arange(6, dtype=torch.float32).view(2, 3)
        with (
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.all_to_all_single",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            unittest.mock.patch(
                "torch.ops._c10d_functional.wait_tensor",
                side_effect=lambda value: value,
            ),
            unittest.mock.patch.object(_DeduplicatedDispatch, "apply") as apply,
            torch.no_grad(),
        ):
            actual = _deduplicated_dispatch_maybe_autograd(
                x,
                torch.tensor([0, 1]),
                torch.tensor([0, 0, 1, 1]),
                torch.tensor([0, 0, 1, 1]),
                [2],
                [2],
                [4],
                [4],
                unittest.mock.MagicMock(),
            )

        torch.testing.assert_close(actual, x[[0, 0, 1, 1]])
        apply.assert_not_called()

    def test_dispatch_split_sizes_match_payloads(self):
        topk_expert_ids = torch.tensor(
            [
                [0, 1, 8, 9, 10, 11, 12, 13],
                [0, 2, 3, 4, 5, 6, 7, 16],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ]
        )
        (
            assignment_counts,
            _,
            _,
            deduplicated_token_indices,
            deduplicated_splits,
            expand_indices,
        ) = _prepare_deduplicated_dispatch(
            topk_expert_ids,
            num_experts=32,
            ep_size=8,
        )

        self.assertEqual(
            int(assignment_counts.sum().item()),
            expand_indices.shape[0],
        )
        self.assertEqual(
            int(deduplicated_splits.sum().item()),
            deduplicated_token_indices.shape[0],
        )
        self.assertLess(
            deduplicated_token_indices.shape[0],
            expand_indices.shape[0],
        )

    def test_backward_matches_standard_assignment_order(self):
        x = torch.arange(6, dtype=torch.float32).view(2, 3).requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        token_indices = torch.tensor([0, 0, 1, 1])
        deduplicated_token_indices = torch.tensor([0, 1])
        expand_indices = torch.tensor([0, 0, 1, 1])
        grad = torch.arange(12, dtype=torch.float32).view(4, 3)

        with (
            unittest.mock.patch(
                "torchtitan.models.common.token_dispatcher.all_to_all_single",
                side_effect=lambda tensor, *args, **kwargs: tensor,
            ),
            unittest.mock.patch(
                "torch.ops._c10d_functional.wait_tensor",
                side_effect=lambda tensor: tensor,
            ),
        ):
            routed = _DeduplicatedDispatch.apply(
                x,
                deduplicated_token_indices,
                expand_indices,
                token_indices,
                [2],
                [2],
                [4],
                [4],
                unittest.mock.MagicMock(),
            )
            routed.backward(grad)

        reference_x[token_indices].backward(grad)
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)

    def test_reconstructs_stable_expert_sorted_tokens(self):
        topk_expert_ids = torch.tensor(
            [
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [0, 2, 1, 3],
                [4, 6, 5, 7],
            ]
        )
        (
            counts,
            _,
            sorted_token_indices,
            deduplicated_token_indices,
            deduplicated_splits,
            expand_indices,
        ) = _prepare_deduplicated_dispatch(
            topk_expert_ids,
            num_experts=8,
            ep_size=2,
        )

        expected_assignment_order = torch.argsort(
            topk_expert_ids.reshape(-1), stable=True
        )
        expected_token_indices = expected_assignment_order // 4
        torch.testing.assert_close(sorted_token_indices, expected_token_indices)
        torch.testing.assert_close(
            counts, torch.bincount(topk_expert_ids.reshape(-1), minlength=8)
        )
        torch.testing.assert_close(deduplicated_splits, torch.tensor([3, 3]))

        rank_starts = deduplicated_splits.cumsum(0) - deduplicated_splits
        destination_ranks = topk_expert_ids.reshape(-1)[expected_assignment_order] // 4
        reconstructed = deduplicated_token_indices[
            rank_starts[destination_ranks] + expand_indices
        ]
        torch.testing.assert_close(reconstructed, expected_token_indices)

    def test_topk_one_has_no_deduplication(self):
        topk_expert_ids = torch.tensor([[0], [3], [4], [7]])
        _, _, _, deduplicated_token_indices, splits, _ = _prepare_deduplicated_dispatch(
            topk_expert_ids,
            num_experts=8,
            ep_size=2,
        )
        torch.testing.assert_close(splits, torch.tensor([2, 2]))
        torch.testing.assert_close(
            deduplicated_token_indices, torch.tensor([0, 1, 2, 3])
        )

    def test_rejects_scoring_before_experts(self):
        with self.assertRaisesRegex(ValueError, "requires score_before_experts=False"):
            DeduplicatedAllToAllTokenDispatcher(
                DeduplicatedAllToAllTokenDispatcher.Config(
                    num_experts=8,
                    top_k=2,
                    score_before_experts=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
