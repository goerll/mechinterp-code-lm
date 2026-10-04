import unittest
from types import SimpleNamespace

import torch

from scripts.final_datasets import shift_boundary, transfer_pairs
from scripts.final_experiments import head_hooks, keep_hooks


class InterventionTests(unittest.TestCase):
    def test_single_head_and_position_are_isolated(self):
        x = torch.zeros(1, 3, 2, 4)
        donor = torch.ones_like(x)
        hook = head_hooks(
            [(0, 1)], {"blocks.0.attn.hook_z": donor}, position="changed", changed=[1]
        )[0][1]
        y = hook(x, None)
        self.assertEqual(y.sum().item(), 4)
        self.assertTrue(torch.equal(y[0, 1, 1], torch.ones(4)))
        self.assertEqual(x.sum().item(), 0)
        self.assertEqual(donor.sum().item(), 24)

    def test_keep_set_freezes_only_complement(self):
        model = SimpleNamespace(cfg=SimpleNamespace(n_layers=1, n_heads=2))
        donor = {
            "blocks.0.attn.hook_z": torch.ones(1, 3, 2, 4),
            "blocks.0.hook_mlp_out": torch.ones(1, 3, 8),
        }
        hooks = keep_hooks(model, [(0, 1)], donor)
        y = hooks[0][1](torch.zeros(1, 3, 2, 4), None)
        self.assertEqual(y[:, :, 0].sum().item(), 12)
        self.assertEqual(y[:, :, 1].sum().item(), 0)
        self.assertEqual(hooks[1][1](None, None).sum().item(), 24)

    def test_transfer_factorial_and_unique_labels(self):
        pairs = transfer_pairs()  # Generator parses both completed variants.
        self.assertEqual(len(pairs), 384)
        self.assertEqual(len({r["label"] for r in pairs}), 384)
        self.assertEqual(
            {r["task"] for r in pairs}, {"increase", "continuation", "dedent"}
        )
        for r in pairs:
            self.assertNotEqual(r["clean_target"], r["corrupt_target"])

    def test_boundary_shift_preserves_reconstructed_source(self):
        def reconstruct(prompt, target):
            prefix, suffix = prompt.removeprefix("<fim-prefix>").split("<fim-suffix>")
            return prefix + target + suffix.removesuffix("<fim-middle>")

        for row in transfer_pairs():
            shifted = shift_boundary(row)
            for side in ("clean", "corrupt"):
                self.assertEqual(
                    reconstruct(row[side + "_prompt"], row[side + "_target"]),
                    reconstruct(shifted[side + "_prompt"], shifted[side + "_target"]),
                )
