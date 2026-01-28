from __future__ import annotations

import argparse
import transformers

from transformers import HfArgumentParser

from .tps import add_tps_parser
from transformers.commands.chat import ChatCommand


def build_parser() -> argparse.ArgumentParser:
    parser = HfArgumentParser(prog="mblt-model-zoo")
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_tps_parser(commands_parser)
    ChatCommand.register_subcommand(commands_parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path_or_address,
        trust_remote_code=True,
    )

    model_type = getattr(config, "model_type", "")

    if model_type.startswith("mobilint-"):
        original_model_type = model_type[len("mobilint-"):]
        arch_name = config.architectures[0] if getattr(config, "architectures", None) else ""

        import importlib
        module = importlib.import_module(
            f"mblt_model_zoo.hf_transformers.models.{original_model_type}.modeling_{original_model_type}"
        )
        setattr(transformers, config.architectures[0], module.__dict__[config.architectures[0]])
        
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        )

        if arch_name.endswith("CausalLM"):
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type] = arch_name
        elif arch_name.endswith("ConditionalGeneration"):
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[model_type] = arch_name

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
