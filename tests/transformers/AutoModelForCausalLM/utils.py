import math
from typing import List

import torch
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from transformers.generation.streamers import BaseStreamer


class BatchTextStreamer(BaseStreamer):
    def __init__(self, tokenizer, batch_size: int = 1, skip_prompt: bool = True):
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

        self.console = Console()
        self.live = Live(
            self.make_table(),
            console=self.console,
            transient=False,
            vertical_overflow="visible",
            auto_refresh=False,
        )

    def put(self, value: torch.Tensor):
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
            self.print_buffers[i] += new_text

            if self.tokenizer.eos_token_id in tokens and value[i].numel() == 1:
                self.finished[i] = True

        self.live.update(self.make_table())

    def end(self):
        if self.live:
            self.live.stop()

    def make_table(self) -> Table:
        table = Table(
            show_header=False,
            header_style="bold magenta",
            box=box.ROUNDED,
            expand=True,
            show_lines=True,
        )

        for _ in range(self.num_columns):
            table.add_column(no_wrap=False)

        outputs: List[str] = []
        states: List[str] = []

        for i in range(self.batch_size):
            status = "DONE" if self.finished[i] else "GEN"
            style = "green" if self.finished[i] else "yellow"

            display_text = self.print_buffers[i]

            outputs.append(display_text)
            states.append(f"[{style}]{status}[/{style}]")

        for i in range(self.num_rows):
            sliced_states = states[i * self.num_columns : (1 + i) * self.num_columns]
            sliced_outputs = outputs[i * self.num_columns : (1 + i) * self.num_columns]
            table.add_row("State", *sliced_states)
            table.add_row("Generated Output", *sliced_outputs)

        return table
