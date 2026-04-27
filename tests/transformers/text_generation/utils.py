import math
import textwrap

import torch
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from transformers.generation.streamers import BaseStreamer


class BatchTextStreamer(BaseStreamer):
    """Stream batch decoding results in a fixed-size rich table."""

    def __init__(
        self,
        tokenizer,
        batch_size: int = 1,
        skip_prompt: bool = True,
        alert_token_length: list[int] = [],
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt

        self.batch_size = batch_size
        self.num_columns = math.ceil(math.sqrt(self.batch_size))
        self.num_rows = math.ceil(batch_size / self.num_columns)
        self.token_cache = [[] for _ in range(self.batch_size)]
        self.text_buffers = [""] * self.batch_size
        self.print_buffers = [""] * self.batch_size
        self.finished = [False] * self.batch_size
        self.next_tokens_are_prompt = True
        self.alert_token_length = alert_token_length

        self.console = Console()
        self.live = Live(
            self.make_table(),
            console=self.console,
            transient=False,
            vertical_overflow="visible",
            auto_refresh=False,
        )

    def put(self, value: torch.Tensor):
        """Append streamed tokens and refresh the live tail view."""
        if self.live and not self.live.is_started:
            self.live.start()

        if value.dim() == 1 or value.shape[0] == 1:
            value = value.view(self.batch_size, -1)

        batch_size, _ = value.shape

        assert batch_size == self.batch_size

        for i in range(batch_size):
            if self.finished[i]:
                continue

            tokens = value[i].tolist()
            self.token_cache[i].extend(tokens)

            text = self.tokenizer.decode(self.token_cache[i], skip_special_tokens=False)

            new_text = text[len(self.text_buffers[i]) :]
            self.text_buffers[i] = text
            self.print_buffers[i] += new_text + (
                f"<{len(self.token_cache[i])} tokens used>"
                if len(self.token_cache[i]) in self.alert_token_length
                else ""
            )

            if self.tokenizer.eos_token_id in tokens and value[i].numel() == 1:
                self.finished[i] = True

        self.live.update(self.make_table(), refresh=True)

    def end(self):
        """Stop live rendering and print the final full table once."""
        if self.live:
            self.live.stop()
            self.console.print()
            self.console.print(self.make_table(truncate=False))

    def _get_cell_width(self) -> int:
        """Estimate the usable width for each table cell."""
        border_width = self.num_columns + 1
        inner_padding = self.num_columns * 2
        usable_width = self.console.size.width - border_width - inner_padding
        return max(8, usable_width // self.num_columns)

    def _get_cell_max_lines(self) -> int:
        """Estimate the maximum visible line count per cell."""
        border_lines = self.num_rows + 1
        live_margin = 3
        usable_height = self.console.size.height - border_lines - live_margin
        return max(1, usable_height // self.num_rows)

    def _wrap_text(self, text: str, width: int) -> list[str]:
        """Wrap text according to the estimated cell width."""
        wrapped_lines: list[str] = []

        for raw_line in text.splitlines():
            if not raw_line:
                wrapped_lines.append("")
                continue

            wrapped_lines.extend(
                textwrap.wrap(
                    raw_line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )

        if text.endswith("\n"):
            wrapped_lines.append("")

        return wrapped_lines or [""]

    def _truncate_for_display(self, text: str, truncate: bool) -> str:
        """Keep the latest visible lines and mark omitted leading content."""
        if not truncate:
            return text

        wrapped_lines = self._wrap_text(text, self._get_cell_width())
        max_lines = self._get_cell_max_lines()

        if len(wrapped_lines) <= max_lines:
            return text

        if max_lines == 1:
            return "..."

        visible_lines = wrapped_lines[-(max_lines - 1) :]
        return "\n".join(["...", *visible_lines])

    def make_table(self, truncate: bool = True) -> Table:
        """Build the rich table for the current batch outputs."""
        table = Table(
            show_header=False,
            header_style="bold magenta",
            box=box.ROUNDED,
            expand=True,
            show_lines=True,
        )

        for _ in range(self.num_columns):
            table.add_column(overflow="fold", no_wrap=False)

        for i in range(self.num_rows):
            sliced_outputs = self.print_buffers[i * self.num_columns : (1 + i) * self.num_columns]
            display_outputs = [self._truncate_for_display(output, truncate) for output in sliced_outputs]
            if len(display_outputs) < self.num_columns:
                display_outputs.extend([""] * (self.num_columns - len(display_outputs)))
            table.add_row(*display_outputs)

        return table
