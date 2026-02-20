"""Prepare prompts for online OFO simulation load generation.

Downloads a HuggingFace dataset and extracts prompts into a JSONL file
that `run_online_ofo.py` reads.  Each line is a JSON object with
`model_label` and `prompt` fields.

Usage:
    python examples/prepare_prompts.py \
        --dataset openai/gsm8k \
        --split test \
        --text-field question \
        --model-labels Llama-3.1-8B Llama-3.1-70B \
        --num-prompts 500 \
        --output data/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("prepare_prompts")


def main(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    logger.info("Loading dataset %s (split=%s)...", args.dataset, args.split)

    ds_args = [args.dataset]
    if args.subset:
        ds_args.append(args.subset)
    ds = load_dataset(*ds_args, split=args.split)

    texts: list[str] = []
    for row in ds:
        text = str(row[args.text_field])
        if text.strip():
            texts.append(text.strip())
        if len(texts) >= args.num_prompts:
            break

    if len(texts) < args.num_prompts:
        logger.warning(
            "Only found %d prompts (requested %d). Using what's available.",
            len(texts),
            args.num_prompts,
        )

    logger.info("Extracted %d prompts", len(texts))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for text in texts:
            for label in args.model_labels:
                f.write(json.dumps({"model_label": label, "prompt": text}) + "\n")

    logger.info(
        "Wrote %d lines to %s (%d prompts x %d models)",
        len(texts) * len(args.model_labels),
        out_path,
        len(texts),
        len(args.model_labels),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare prompts for online OFO simulation")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g. openai/gsm8k)")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name (e.g. main)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--text-field", default="question", help="Field name containing the prompt text")
    parser.add_argument(
        "--model-labels",
        nargs="+",
        required=True,
        help="Model labels to generate prompts for",
    )
    parser.add_argument("--num-prompts", type=int, default=500, help="Number of prompts to extract")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
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

    main(args)
