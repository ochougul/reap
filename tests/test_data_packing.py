import random

import torch
from datasets import Dataset

from reap.data import BaseDatasetProcessor


class _Tokenizer:
    pad_token_id = 0


class _PackedProcessor(BaseDatasetProcessor):
    category_field = None

    @staticmethod
    def _map_fn(sample):
        return sample

    def _encode_sample(self, sample):
        return torch.full((1, sample["length"]), sample["token"], dtype=torch.long)


def test_packed_nontruncating_mode_carries_samples_to_the_next_sequence():
    processor = _PackedProcessor(
        dataset=Dataset.from_dict({"length": [3, 3, 6], "token": [1, 2, 3]}),
        tokenizer=_Tokenizer(),
        max_input_len=5,
        batch_size=1,
        split_by_category=False,
        truncate=False,
    )

    random.seed(42)
    batches = processor.get_processed_dataset(batches_per_category=10)["all"]

    assert len(batches) == 2
    assert sum(int(batch["attention_mask"].sum()) for batch in batches) == 6
    assert {
        int(token)
        for batch in batches
        for token, mask in zip(batch["input_ids"].flatten(), batch["attention_mask"].flatten())
        if mask
    } == {1, 2}
    assert processor.packing_stats["all"] == {
        "source_rows": 3,
        "eligible_source_rows": 2,
        "eligible_source_tokens": 6,
        "skipped_source_rows_over_max_length": 1,
        "skipped_source_tokens_over_max_length": 6,
        "packed_sequences": 2,
    }
