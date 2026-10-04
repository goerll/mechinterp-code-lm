from scripts.indentation_dataset import generate_indentation_pairs


def test_factorial_size_and_split():
    pairs = generate_indentation_pairs()
    assert len(pairs) == 128
    assert sum(p.split == "discovery" for p in pairs) == 64
    assert sum(p.split == "confirmation" for p in pairs) == 64


def test_targets_and_prompts_are_matched():
    for p in generate_indentation_pairs():
        assert p.clean_target_spaces == p.clean_statement_spaces + 4
        assert p.corrupt_target_spaces == p.corrupt_statement_spaces + 4
        assert p.clean_target_spaces != p.corrupt_target_spaces
        assert p.clean_prompt != p.corrupt_prompt
        assert p.clean_prompt != p.negative_control_prompt
