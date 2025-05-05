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
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    print(f"⏱️ [Setup] {time.time() - time0:.3f}s")

    time1 = time.time()
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    print(f"⏱️ [Token preparation] {time.time() - time1:.3f}s")

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    time2 = time.time()
    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            outputs = model(input_ids=tokens[:, :cur_pos])
            logits = outputs.logits

        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).reshape(-1)

        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token

        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)

        prev_pos = cur_pos
        if is_finished.all():
            break
    print(f"\n⏱️ [Generation loop] {time.time() - time2:.3f}s")

    time3 = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"⏱️ [GC + CUDA cleanup] {time.time() - time3:.3f}s")

    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    time4 = time.time()
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]

            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]

            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            reward_start = time.time()
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            print(f"⏱️ [Reward {idx}] {time.time() - reward_start:.3f}s")

            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    print(f"⏱️ [Episode assembly] {time.time() - time4:.3f}s")

    print("\r", end=" " * 100, flush=True)
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
