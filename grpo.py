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


def get_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding behavior
    return tokenizer


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
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    bsz = len(batch.prefix) * num_answer_per_question

    print(f"[Rollout] Batch size: {bsz}, Max generation length: {max_gen_len}")

    prefix_texts = [prefix for prefix in batch.prefix for _ in range(num_answer_per_question)]
    encoded = tokenizer(prefix_texts, padding=True, return_tensors="pt", return_attention_mask=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model = model.to(device).eval()

    with torch.inference_mode():
        print("[Rollout] Starting generation...")
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_gen_len,
            do_sample=True,
            top_k=50,
            pad_token_id=pad_token_id,
            eos_token_id=end_token_id,
        )
        print("[Rollout] Generation completed.")

    gc.collect()
    torch.cuda.empty_cache()

    episodes = []
    generated_ids = generated_ids.tolist()
    input_ids_list = input_ids.tolist()

    for i in range(bsz):
        prompt_len = len(input_ids_list[i])
        generated_token_ids = generated_ids[i][prompt_len:]

        if pad_token_id in generated_token_ids:
            generated_token_ids = generated_token_ids[:generated_token_ids.index(pad_token_id)]

        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        print(f"[Rollout] Sample {i+1}/{bsz} Generated Text: {generated_text[:100]}...")

        base_idx = i // num_answer_per_question
        rewards = reward_function(
            response=generated_text,
            sentence=batch.sentences[base_idx],
            end_token=end_token,
        )
        print(f"[Rollout] Rewards: {rewards}")


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

    print(f"[Rollout] Completed {len(episodes)} episodes in {time.time() - total_start:.2f} seconds.")
    return episodes



def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    start_time = time.time()
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)

    output = []
    for group in groups.values():
        rewards = [ep.reward for ep in group]
        mean, std = np.mean(rewards), np.std(rewards)

        # If all rewards are the same or nearly so, avoid division
        if std < 1e-6:
            for ep in group:
                norm_reward = ep.reward - mean  # or just 0.0
                output.append(dataclasses.replace(ep, reward=norm_reward))
        else:
            for ep in group:
                norm_reward = (ep.reward - mean) / (std + 1e-4)
                output.append(dataclasses.replace(ep, reward=norm_reward))

    return output



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
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
    print("[Update Policy] Normalizing rewards...")
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))

    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(ep.generated_token_ids) for ep in episodes)

    print(f"[Update Policy] Number of episodes: {len(episodes)}")
    print(f"[Update Policy] Micro batch size: {micro_batch_size}")
    print(f"[Update Policy] Number of micro-batches: {num_micro_batches}")
    print(f"[Update Policy] Total target tokens: {num_target_tokens}")

    entropy = 0.0
    model.train()  # ✅ ensure model is in training mode

    for i in range(0, len(episodes), micro_batch_size):
        j = min(i + micro_batch_size, len(episodes))
        batch = episodes[i:j]
        lens = [len(ep.prefix_token_ids) + len(ep.generated_token_ids) for ep in batch]
        max_len = max(lens)

        token_ids = [
            ep.prefix_token_ids + ep.generated_token_ids + [pad_token_id] * (max_len - l)
            for ep, l in zip(batch, lens)
        ]
        masks = [
            [0]*len(ep.prefix_token_ids) + [1]*len(ep.generated_token_ids) + [0]*(max_len - l)
            for ep, l in zip(batch, lens)
        ]
        advantages = [ep.reward for ep in batch]

        print(f"[Update Policy] Processing micro-batch {i // micro_batch_size + 1}/{num_micro_batches}")
        print(f"  - Max sequence length in batch: {max_len}")
        print(f"  - Advantage stats: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}")

        token_ids = torch.tensor(token_ids, device=device)
        masks = torch.tensor(masks, device=device, dtype=torch.bool)
        advantages = torch.tensor(advantages, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            target_masks = masks[:, 1:]

            logits = model(input_ids).logits  # ✅ No .float(), no .detach()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            target_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none"
        ).reshape(input_ids.size(0), -1)

        with torch.no_grad():
            batch_entropy = (compute_entropy(logits) * target_masks).sum() / num_target_tokens
            entropy += batch_entropy
            print(f"  - Entropy (avg per token): {batch_entropy.item():.4f}")

        obj = log_probs * advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens

        loss = -obj
        print(f"  - Loss: {loss.item():.4f}")

        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    print(f"[Update Policy] Gradient norm after clipping: {grad_norm.item():.4f}")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    total_time = time.time() - total_start
    print(f"[Update Policy] Policy update completed in {total_time:.2f} seconds.")

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

