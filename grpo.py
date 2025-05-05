import dataclasses
import gc
import math
import time
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_types import Episode, MiniBatch

@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    batch: MiniBatch,
    tokenizer: AutoTokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    total_start = time.time()

    time0 = time.time()
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    print(f"⏱️ [Setup] {time.time() - time0:.3f}s")

    # Prepare input tokens
    time1 = time.time()
    input_ids = []
    for t in prefix_token_ids:
        for _ in range(num_answer_per_question):
            input_ids.append(torch.tensor(t, dtype=torch.long))
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    input_ids = input_ids.to(device)
    print(f"⏱️ [Token preparation] {time.time() - time1:.3f}s")

    # Generate sequences
    time2 = time.time()
    attention_mask = input_ids != pad_token_id
    model = model.to(device).eval()
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_gen_len,
        do_sample=True,
        top_k=50,
        pad_token_id=pad_token_id,
        eos_token_id=end_token_id,
    )
    print(f"⏱️ [Generation] {time.time() - time2:.3f}s")

    # Clear GPU memory
    time3 = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"⏱️ [GC + CUDA cleanup] {time.time() - time3:.3f}s")

    # Convert to episodes
    time4 = time.time()
    episodes = []
    generated_ids = generated_ids.tolist()
    input_ids_list = input_ids.tolist()

    for i in range(bsz):
        prompt_len = len(input_ids_list[i])
        generated_token_ids = generated_ids[i][prompt_len:]

        if pad_token_id in generated_token_ids:
            generated_token_ids = generated_token_ids[:generated_token_ids.index(pad_token_id)]

        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        reward_start = time.time()
        base_idx = i // num_answer_per_question
        rewards = reward_function(
            response=generated_text,
            numbers=batch.numbers[base_idx],
            target=batch.target[base_idx],
            end_token=end_token,
        )
        print(f"⏱️ [Reward {i}] {time.time() - reward_start:.3f}s")

        episode = Episode(
            prefix=batch.prefix[base_idx],
            text=batch.prefix[base_idx] + generated_text,
            prefix_token_ids=batch.prefix_token_ids[base_idx],
            prefix_tokens=batch.prefix_tokens[base_idx],
            generated_token_ids=generated_token_ids,
            is_finished=(end_token_id in generated_token_ids),
            reward=rewards["reward"],
            reward_info=rewards["reward_info"],
        )
        episodes.append(episode)

    print(f"⏱️ [Episode assembly] {time.time() - time4:.3f}s")
    print(f"✅ [Total rollout time] {time.time() - total_start:.3f}s")
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    start_time = time.time()
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)

    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)

    print(f"⏱️ [Reward normalization] {time.time() - start_time:.3f}s")
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    start_time = time.time()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    print(f"⏱️ [Entropy computation] {time.time() - start_time:.3f}s")
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    total_start = time.time()
    episodes = normalize_rewards_per_group(episodes)

    sort_start = time.time()
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    print(f"⏱️ [Sort for batching] {time.time() - sort_start:.3f}s")

    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    batch_start = time.time()
    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            forward_start = time.time()
            logits = model.forward(input_token_ids).float()
            print(f"⏱️ [Forward pass] {time.time() - forward_start:.3f}s")

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            entropy += (compute_entropy(logits) * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()
    print(f"\n⏱️ [Total batch processing] {time.time() - batch_start:.3f}s")

    update_start = time.time()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"⏱️ [Optimizer update] {time.time() - update_start:.3f}s")
    print(f"✅ [Total policy update] {time.time() - total_start:.3f}s")

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
