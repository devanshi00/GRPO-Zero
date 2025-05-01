import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
   "Using the sentence: "{sentence}", which contains gender bias (annotated as [1] for male bias and [0] for female bias), rewrite the sentence to mitigate this bias while preserving the original meaning and intent. "
   "Explain your reasoning in <think> </think> tags. "
   "Return the debiased version of the sentence in <answer> </answer> tags." 
   "For example: <answer> The person excelled in their field regardless of gender. </answer> "
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class CountdownTasksDataset(Dataset):
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "data")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


# def answer_reward_function(
#     response: str, numbers: List[int] = None, target: int = None
# ) -> float:
#     """
#     Checks if the answer uses all numbers exactly once and evaluates to the target
#     """
#     answer_regex = r"<answer>(.*?)<\/answer>"
#     answer_match = re.search(answer_regex, response, re.DOTALL)
#     if not answer_match:
#         return 0.0

#     answer_content = answer_match.group(1)
#     if not answer_content:
#         return 0.0

#     allowed_chars = r"^[0-9+\-*/() ]+$"
#     if not re.match(allowed_chars, answer_content):
#         return 0.0

#     # Check if the answer uses all numbers exactly once
#     used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
#     if sorted(used_numbers) != sorted(numbers):
#         return 0.0

#     # Check if the answer evaluates to the target
#     try:
#         result = eval(answer_content, {"__builtins__": None}, {})
#         if abs(float(result) - float(target)) < 1e-5:
#             return 1.0
#     except:
#         pass

#     return 0.0

def answer_reward_function(
    response: str, input_sentence: str = ""
) -> float:
    """
    Penalizes if response contains 'she' or 'he', otherwise gives reward of +1 for each match of input_sentence.
    """
    # Penalize if gendered pronouns are present
    if re.search(r"\b(she|he)\b", response, re.IGNORECASE):
        print("hi")
        return -10.0
    seqs = [response.split(" "), input_sentence.split(" ")]
    matches = 0.0
    m1={}
    m2={}
    if len(seqs[0]) < len(seqs[1]):
        seqs[0] += [" "] * (len(seqs[1]) - len(seqs[0]))
    else:
        seqs[1] += [" "] * (len(seqs[1]) - len(seqs[0]))
        
    
    for word1, word2 in zip(seqs[0], seqs[1]):
        if m1.get(word1):
            m1[word1]+=1
        if m2.get(word2):
            m2[word2]+=1
        else:
            m1[word1] = 1
            m2[word2]=1
        
    # min_len = min(len(m1), len(seqs[m2]))
    
    shared_items = {k: m1[k] for k in m1 if k in m2 and m1[k] == m2[k]}
    for i in shared_items:
        print(i)
    k=len(shared_items)
        
    # matches = sum(response.split(" ") == input_sentence.split(" "))
    # print(matches)
    return float(k)
    


    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
