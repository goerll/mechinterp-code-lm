from dataclasses import fields
import unittest

from scripts.bracket_dataset import BracketPair, dataset_fingerprint, generate_bracket_pairs


class BracketDatasetTests(unittest.TestCase):
    def setUp(self):
        self.pairs = generate_bracket_pairs()

    def test_expected_size_and_splits(self):
        self.assertEqual(len(self.pairs), 192)
        self.assertEqual(sum(pair.split == "discovery" for pair in self.pairs), 96)
        self.assertEqual(sum(pair.split == "confirmation" for pair in self.pairs), 96)

    def test_clean_and_corrupt_differ_only_at_target_opener(self):
        for pair in self.pairs:
            differences = [
                index
                for index, (clean, corrupt) in enumerate(
                    zip(pair.clean_prompt, pair.corrupt_prompt)
                )
                if clean != corrupt
            ]
            self.assertEqual(differences, [pair.target_char_index])
            self.assertEqual(pair.clean_prompt[pair.target_char_index], pair.clean_opener)
            self.assertEqual(pair.corrupt_prompt[pair.target_char_index], pair.corrupt_opener)

    def test_confirmation_uses_unseen_scaffolds(self):
        discovery = {pair.template_id for pair in self.pairs if pair.split == "discovery"}
        confirmation = {pair.template_id for pair in self.pairs if pair.split == "confirmation"}
        self.assertTrue(discovery.isdisjoint(confirmation))

    def test_labels_and_fingerprint_are_deterministic(self):
        regenerated = generate_bracket_pairs()
        self.assertEqual([pair.label for pair in self.pairs], [pair.label for pair in regenerated])
        self.assertEqual(dataset_fingerprint(self.pairs), dataset_fingerprint(regenerated))

    def test_serialized_schema_contains_every_field(self):
        expected = {field.name for field in fields(BracketPair)}
        self.assertEqual(set(self.pairs[0].to_dict()), expected)


if __name__ == "__main__":
    unittest.main()
