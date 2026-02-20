"""Build pre-formatted OpenAI Chat Completion request dicts for online simulation.

Loads prompts from one of two hard-coded datasets (GPQA extended, LM Arena Chat)
and writes JSONL files with complete request dicts ready to send to vLLM.

The `OnlineDatacenter` loads these pre-built payloads instead of constructing
them at runtime, separating data preparation from load generation.

Requires: `pip install datasets openai`

Usage:
    python data/online/build_requests.py \
        --dataset lm-arena-chat \
        --config examples/online/online_config.example.json \
        --out-dir data/online/requests/ \
        --num-requests 1000

    python data/online/build_requests.py \
        --dataset gpqa \
        --config examples/online/online_config.example.json \
        --out-dir data/online/requests/ \
        --num-requests 500
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

logger = logging.getLogger("build_requests")

# Per-model extra_body and system_prompt overrides.
# Some models (e.g., Qwen3, DeepSeek) need `chat_template_kwargs` to disable
# "thinking" mode for standard chat workloads; others (e.g., Nemotron) need a
# system prompt. Add entries here keyed by HuggingFace model ID.
#
# extra_body entries are merged into every request dict for the matching model.
MODEL_EXTRA_BODY: dict[str, dict] = {
    "Qwen/Qwen3-8B": {"chat_template_kwargs": {"enable_thinking": False}},
    "Qwen/Qwen3-14B": {"chat_template_kwargs": {"enable_thinking": False}},
    "Qwen/Qwen3-32B": {"chat_template_kwargs": {"enable_thinking": False}},
}
MODEL_SYSTEM_PROMPT: dict[str, str] = {}
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def _sample_lm_arena_chat(
    num_requests: int,
    seed: int = 0,
) -> list[str | list[str]]:
    """Sample multi-turn chat prompts from LM Arena Human Preference dataset.

    Each conversation item may yield multiple prompts (one per turn).
    Conversations are shuffled with the given seed.

    Args:
        num_requests: Number of prompts to sample.
        seed: Random seed for dataset shuffling.

    Returns:
        List of prompts. Single-turn prompts are strings, multi-turn are
        lists of alternating user/assistant message strings.
    """
    data = load_dataset(
        "lmarena-ai/arena-human-preference-100k",
        split="train",
    ).shuffle(seed=seed)

    prompts: list[str | list[str]] = []
    for item in data:
        num_turns = item["turn"]
        conversation = item["conversation_a"]

        for turns in range(num_turns):
            if len(prompts) >= num_requests:
                break

            messages: list[str] = []
            for message in conversation[: 2 * turns + 1]:
                messages.append(message["content"])

            prompts.append(messages if len(messages) > 1 else messages[0])

        if len(prompts) >= num_requests:
            break

    _maybe_oversample(prompts, num_requests, seed)
    return prompts


def _sample_gpqa(
    num_requests: int,
    seed: int = 0,
) -> list[str | list[str]]:
    """Sample multiple-choice science questions from the GPQA extended dataset.

    Choice order is randomized per question.

    Args:
        num_requests: Number of prompts to sample.
        seed: Random seed for dataset shuffling and choice ordering.

    Returns:
        List of prompt strings.
    """
    data = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_extended",
        split="train",
        streaming=True,
    ).shuffle(seed=seed)

    random.seed(seed)

    prompts: list[str | list[str]] = []
    for item in data:
        if len(prompts) >= num_requests:
            break

        choices = [
            item["Incorrect Answer 1"].strip(),
            item["Incorrect Answer 2"].strip(),
            item["Incorrect Answer 3"].strip(),
            item["Correct Answer"].strip(),
        ]
        random.shuffle(choices)

        question = item["Question"]
        prompt = f"What is the correct answer to the following question: {question}\n\nChoices:"
        for letter, choice in zip("ABCD", choices, strict=True):
            prompt += f"\n({letter}) {choice}"

        prompts.append(prompt)

    _maybe_oversample(prompts, num_requests, seed)
    return prompts


def _maybe_oversample(
    items: list[str | list[str]],
    target: int,
    seed: int,
) -> None:
    """Oversample items in-place if fewer than target."""
    if len(items) >= target:
        return
    rng = random.Random(seed)
    original = list(items)
    while len(items) < target:
        items.append(rng.choice(original))


DATASET_SAMPLERS = {
    "lm-arena-chat": _sample_lm_arena_chat,
    "gpqa": _sample_gpqa,
}


def _text_part(text: str) -> ChatCompletionContentPartTextParam:
    """Build a text content part."""
    return ChatCompletionContentPartTextParam(type="text", text=text)


def _prompt_to_messages(prompt: str | list[str]) -> list[ChatCompletionMessageParam]:
    """Convert a prompt to OpenAI Chat Completion messages.

    Uses the content-parts format (`[{"type": "text", "text": ...}]`) for
    future multimodal extensibility.

    Follows the mlenergy benchmark convention:
    - Single string prompt: one user message.
    - Multi-turn (list of strings): alternating user/assistant messages.

    Args:
        prompt: A single prompt string or list of conversation turns.

    Returns:
        A list of typed message params for the Chat Completion API.
    """
    if isinstance(prompt, str):
        return [ChatCompletionUserMessageParam(role="user", content=[_text_part(prompt)])]

    msgs: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(role="user", content=[_text_part(prompt[0])])
    ]
    for i, turn in enumerate(prompt[1:]):
        if i % 2 == 0:
            msgs.append(ChatCompletionAssistantMessageParam(role="assistant", content=[_text_part(turn)]))
        else:
            msgs.append(ChatCompletionUserMessageParam(role="user", content=[_text_part(turn)]))
    return msgs


def build_request_template(
    *,
    model_id: str,
    max_completion_tokens: int,
    system_prompt: str | None = None,
    extra_body: dict | None = None,
) -> CompletionCreateParamsStreaming:
    """Build a request template with all fields except per-request messages.

    The template is a `CompletionCreateParamsStreaming` with `stream` and
    `stream_options` pre-filled. If a system prompt is provided, it is
    included in the template's message list so it is prepended to every
    request.

    Args:
        model_id: Model ID as served by vLLM.
        max_completion_tokens: Maximum output tokens.
        system_prompt: Optional system prompt prepended to every request.
        extra_body: Optional extra fields merged into the request body.

    Returns:
        A typed request template dict.
    """
    system_msgs: list[ChatCompletionMessageParam] = []
    if system_prompt:
        system_msgs.append(ChatCompletionSystemMessageParam(role="system", content=system_prompt))

    template = CompletionCreateParamsStreaming(
        model=model_id,
        messages=system_msgs,
        max_completion_tokens=max_completion_tokens,
        stream=True,
        stream_options={"include_usage": True, "continuous_usage_stats": True},
    )
    if extra_body:
        template.update(extra_body)
    return template


def build_request(
    template: CompletionCreateParamsStreaming,
    prompt: str | list[str],
) -> dict:
    """Build a complete request dict by filling messages into a template.

    Args:
        template: Request template from `build_request_template`.
        prompt: User prompt text or list of conversation turns.

    Returns:
        A dict ready to serialize as JSON to /v1/chat/completions.
    """
    request = dict(template)
    request["messages"] = list(template["messages"]) + _prompt_to_messages(prompt)
    return request


def main(args: argparse.Namespace) -> None:
    dataset_name = args.dataset
    sampler = DATASET_SAMPLERS[dataset_name]

    with open(args.config) as f:
        config = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = config["models"]
    num_requests = args.num_requests
    max_completion_tokens = args.max_completion_tokens
    seed = args.seed

    for model_cfg in models:
        label = model_cfg["model_label"]
        model_id = model_cfg["model_name"]

        logger.info(
            "Sampling %d %s prompts for %s (%s)...",
            num_requests,
            dataset_name,
            label,
            model_id,
        )
        prompts = sampler(num_requests=num_requests, seed=seed)
        logger.info("  Got %d prompts", len(prompts))

        template = build_request_template(
            model_id=model_id,
            max_completion_tokens=max_completion_tokens,
            system_prompt=MODEL_SYSTEM_PROMPT.get(model_id, DEFAULT_SYSTEM_PROMPT),
            extra_body=MODEL_EXTRA_BODY.get(model_id),
        )

        out_path = out_dir / f"{label}.jsonl"
        count = 0
        with open(out_path, "w") as f:
            for prompt in prompts:
                body = build_request(template, prompt)
                f.write(json.dumps(body) + "\n")
                count += 1

        logger.info("Wrote %d requests for %s to %s", count, label, out_path)

    logger.info("Done. Requests written to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build pre-formatted request dicts for online simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_SAMPLERS.keys()),
        help="Dataset to sample from: 'gpqa' (GPQA extended) or 'lm-arena-chat' (LM Arena Chat).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config file (same format as online_config.example.json). "
        "Uses 'model_name' (HF ID) and 'model_label' for output file names.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for per-model JSONL request files.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of requests to sample per model (default: 1000).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum output tokens per request (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset shuffling and oversampling (default: 0).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(args)
