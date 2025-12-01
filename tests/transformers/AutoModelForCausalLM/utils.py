from typing import List

import torch
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from transformers.generation.streamers import BaseStreamer


class BatchTextStreamer(BaseStreamer):
    def __init__(self, tokenizer, request_ids: List[str], skip_prompt: bool = True):
        self.tokenizer = tokenizer
        self.request_ids = request_ids
        self.skip_prompt = skip_prompt

        self.batch_size = len(request_ids)
        self.token_cache = [[] for _ in range(self.batch_size)]
        self.text_buffers = [""] * self.batch_size
        self.print_buffers = [""] * self.batch_size
        self.finished = [False] * self.batch_size
        self.next_tokens_are_prompt = True

        self.console = Console()
        self.live = Live(
            self.make_table(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )

    def put(self, value: torch.Tensor):
        if self.live and not self.live.is_started:
            self.live.start()

        if value.dim() == 1:
            value = value.view(1, -1)

        batch_size, seq_len = value.shape

        if batch_size != self.batch_size:
            return

        for i in range(batch_size):
            if self.finished[i]:
                continue

            tokens = value[i].tolist()
            self.token_cache[i].extend(tokens)

            text = self.tokenizer.decode(self.token_cache[i], skip_special_tokens=True)

            new_text = text[len(self.text_buffers[i]) :]
            self.text_buffers[i] = text
            self.print_buffers[i] += new_text

            if self.tokenizer.eos_token_id in tokens:
                self.finished[i] = True

        self.live.update(self.make_table())

    def end(self):
        if self.live:
            self.live.stop()

    def make_table(self) -> Table:
        table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED, expand=True
        )
        table.add_column("Req ID", style="cyan", width=12, no_wrap=True)
        table.add_column("Generated Output", style="white")
        table.add_column("State", width=8, justify="center")

        for i, rid in enumerate(self.request_ids):
            status = "DONE" if self.finished[i] else "GEN"
            style = "green" if self.finished[i] else "yellow"

            display_text = self.print_buffers[i].replace("\n", "⏎ ")

            table.add_row(rid, display_text, f"[{style}]{status}[/{style}]")
        return table
