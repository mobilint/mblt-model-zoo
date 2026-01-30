from __future__ import annotations

from transformers import HfArgumentParser

from .tps import add_tps_parser


def build_parser() -> HfArgumentParser:
    parser = HfArgumentParser(prog="mblt-model-zoo")
    commands_parser = parser.add_subparsers(help="mblt-model-zoo command helpers")

    add_tps_parser(commands_parser)
    _set_subparser_help(commands_parser, "chat", "Transformers chat interface")

    return parser


def _set_subparser_help(subparsers, name: str, help_text: str) -> None:
    choices_actions = getattr(subparsers, "_choices_actions", [])
    for action in choices_actions:
        if getattr(action, "dest", None) == name:
            action.help = help_text
            return
    pseudo_action_cls = getattr(type(subparsers), "_ChoicesPseudoAction", None)
    if pseudo_action_cls is not None:
        choices_actions.append(pseudo_action_cls(name, [], help_text))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "_handler"):
        exit(args._handler(args))

    parser.print_help()
    exit(1)


if __name__ == "__main__":
    main()
