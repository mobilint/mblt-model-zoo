"""Compatibility bridge for ``mblt-melotts-download``: runs ``melotts-mblt download``."""

from __future__ import annotations


def main() -> int:
    from .._deprecation import print_cli_notice
    from ..cli._melotts import exit_missing_dependency, load_standalone

    print_cli_notice("mblt-melotts-download", "melotts-mblt", "melotts-mblt download")

    download = load_standalone("cli.download")
    if download is None:
        return exit_missing_dependency()
    return download.run_download()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
