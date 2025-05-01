import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

USER_TEMPLATE = (
    'Using the sentence: "{sentence}", which contains gender bias (annotated as [1] for male bias and [0] for female bias), '
    "rewrite the sentence to mitigate this bias while preserving the original meaning and intent. "
    "Explain your reasoning in <think> </think> tags. "
    "Return the debiased version of the sentence in <answer> </answer> tags. "
    "For example: <answer> The person excelled in their field regardless of gender. </answer> "
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class Genderdebiasing(Dataset):
    """Prepare gender-debiasing tasks for training"""

    def __init__(self, tokenizer: Tokenizer, data_path: str, split: str = "train", test_size: int = 100):
        data = pd.read_csv(Path(data_path) / "winogender_dataset.csv")["sent"]  # Keep only 'sent' column
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        self.data = train_data if split == "train" else test_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]
        item = {"sentence": sentence}
        item.update(self.encode_prefix(sentence))
        return item

    def encode_prefix(self, sentence: str):
        """Prefix is the input prompt for the model."""
        user_message = USER_TEMPLATE.format(sentence=sentence)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        if not prefix.strip():
            raise ValueError("Prefix prompt is empty. Check sentence or USER_TEMPLATE.")

        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        sentences = [item["sentence"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            sentences=sentences,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Checks if the response follows the format <think>...</think><answer>...</answer>"""
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"
    if re.match(full_format_regex, response, re.DOTALL):
        return 1.0

    reward = 0.0
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += 0.1
    if re.search(r"<answer>.*?</answer>", response, re.DOTALL):
        reward += 0.5
    return reward


def answer_reward_function(response: str, input_sentence: str = "") -> float:
    """Penalizes for gendered pronouns, rewards word matches with input sentence."""
    # Expanded list of gendered pronouns
    if re.search(r"\b(she|he|him|her|his|hers)\b", response, re.IGNORECASE):
        return -10.0

    words_response = response.split()
    words_input = input_sentence.split()

    word_count_resp = {}
    word_count_input = {}
    for w in words_response:
        word_count_resp[w] = word_count_resp.get(w, 0) + 1
    for w in words_input:
        word_count_input[w] = word_count_input.get(w, 0) + 1

    shared_items = {
        w: min(word_count_resp[w], word_count_input[w]) for w in word_count_resp if w in word_count_input
    }
    return float(sum(shared_items.values()))


def reward_function(
    response: str,
    sentence: str = "",
    end_token: str = None,
) -> Dict[str, Any]:
    """Total reward = 0.1 * format_reward + answer_reward"""
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, sentence)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
