"""
Define Dataset objects used by pytorch models.
"""
from typing import Dict, Sequence

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


class TweetDataset(Dataset):
    """Dataset designed for tweet task"""

    def __init__(
        self,
        texts: Sequence[list[str]],
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        labels: Sequence[list[int]],
    ) -> None:
        """
        Init function
        Parameters
        ----------
        texts: Sequence[list[str]]
            List of tokenized text. The sentence is tokenized into a list of token.
        tokenizer: PreTrainedTokenizer
            Usually a pretrained tokenizer from HuggingFace
        max_len: int
            the max len of the list of tokens
        labels: Sequence[int]
            The corresponding tag of each token
        """
        self.texts: Sequence[list[str]] = texts
        self.labels: Sequence[list[int]] = labels
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_len: int = max_len

    def __len__(self) -> int:
        """Get len of dataset"""
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Get item at index"""
        text = self.texts[index]

        encoding = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"],
            "labels": self.labels[index],
        }

    def get_data_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Get data loader from dataset"""
        data_loader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        return DataLoader(self, **data_loader_params)
