"""Live terminal dashboard for recording sessions."""

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from .config import CollectConfig
from .validator import Severity, validate_clip


class Dashboard:
    """Rich-based live terminal UI for recording."""

    def __init__(self, cfg: CollectConfig):
        self.cfg = cfg
        self.console = Console()
        self._elapsed = 0.0
        self._positive_count = 0
        self._negative_count = 0
        self._holding = False
        self._hold_kind = "pos"
        self._last_clip_info = ""
        self._live = None
        self._flash_color = None  # None, "go", "stop"

    def set_flash(self, color):
        """Set flash color for auto mode: 'go', 'stop', or None."""
        self._flash_color = color

    def print_banner(self):
        """Print startup info before recording begins."""
        self.console.print()
        self.console.print("[bold]Wake Word Collection[/bold]")
        self.console.print(f"  Phrase: [cyan]{self.cfg.phrase}[/cyan]")
        self.console.print(f"  Target: {self.cfg.target_positive} positive samples")
        self.console.print(f"  Output: {self.cfg.output_dir}")
        if self.cfg.auto_capture:
            self.console.print(
                f"  Mode: [bold green]AUTO[/bold green]"
                f" (capture {self.cfg.auto_duration_s}s every {self.cfg.auto_interval_s}s)"
            )
            self.console.print()
            self.console.print("[dim]Controls:[/dim]")
            self.console.print("  [bold]Q[/bold] — stop recording")
            self.console.print()
            self.console.print(
                f"[bold]Get into position. First capture in {self.cfg.auto_interval_s}s.[/bold]"
            )
        else:
            self.console.print(
                f"  Keypress trim: {self.cfg.keypress_trim_start_ms}ms start"
                f" / {self.cfg.keypress_trim_end_ms}ms end"
            )
            self.console.print()
            self.console.print("[dim]Controls:[/dim]")
            self.console.print("  [bold]SPACE[/bold] — toggle positive (press to start, press to stop)")
            self.console.print("  [bold]N[/bold]     — toggle negative (press to start, press to stop)")
            self.console.print("  [bold]R[/bold]     — undo last mark / cancel current")
            self.console.print("  [bold]Q[/bold]     — stop recording")
            self.console.print()
            self.console.print("[dim]Gaps between marks are also saved as negative audio.[/dim]")
        self.console.print()

    def _build_display(self) -> Panel:
        """Build the live display panel."""
        minutes = int(self._elapsed) // 60
        seconds = int(self._elapsed) % 60

        filled = min(self._positive_count, self.cfg.target_positive)
        total = self.cfg.target_positive

        # Flash states (auto mode) — large simple display visible from distance
        if self._flash_color == "go":
            lines = [
                "",
                "        [bold white]▶▶▶  SPEAK NOW  ◀◀◀[/bold white]",
                "",
                f"        [white]Captured: {filled}/{total}[/white]",
                "",
            ]
            return Panel(
                "\n".join(lines),
                title="Wake Word Collection",
                border_style="bold white",
                style=Style(bgcolor="green"),
            )

        if self._flash_color == "stop":
            lines = [
                "",
                "        [bold white]■■■    STOP    ■■■[/bold white]",
                "",
                f"        [white]Captured: {filled}/{total}[/white]",
                "",
            ]
            return Panel(
                "\n".join(lines),
                title="Wake Word Collection",
                border_style="bold white",
                style=Style(bgcolor="red"),
            )

        # Normal display (manual mode or auto waiting)
        bar_width = 30
        filled_chars = int(bar_width * filled / total) if total > 0 else 0
        bar = (
            "[green]" + "#" * filled_chars + "[/green]"
            + "[dim]" + "-" * (bar_width - filled_chars) + "[/dim]"
        )

        if self._holding and self._hold_kind == "neg":
            status = "[bold red]● REC NEGATIVE[/bold red]"
        elif self._holding:
            status = "[bold green]● REC POSITIVE[/bold green]"
        elif self.cfg.auto_capture:
            status = "[dim]Waiting... (Q to quit)[/dim]"
        else:
            status = "[dim]Listening (SPACE=pos, N=neg)[/dim]"

        lines = [
            f'Phrase: [cyan]"{self.cfg.phrase}"[/cyan]',
            f"Elapsed: {minutes:02d}:{seconds:02d}    {status}",
            "",
            f"Positive:  {bar}  {filled}/{total}",
            f"Negative:  [red]{self._negative_count}[/red] explicit",
        ]

        if self._last_clip_info:
            lines.append("")
            lines.append(self._last_clip_info)

        return Panel("\n".join(lines), title="Wake Word Collection", border_style="blue")

    def update_tick(self, elapsed: float):
        """Called on each tick to update elapsed time."""
        self._elapsed = elapsed

    def update_hold_start(self, kind: str = "pos"):
        self._holding = True
        self._hold_kind = kind

    def update_hold_end(self, hold, clip_audio: np.ndarray):
        """Called when a hold completes. Validates and shows feedback."""
        self._holding = False

        if hold.kind == "neg":
            self._negative_count += 1
            duration_s = len(clip_audio) / self.cfg.sample_rate
            self._last_clip_info = (
                f"Last clip: [red]{duration_s:.1f}s, NEGATIVE[/red]"
            )
            return

        self._positive_count += 1

        report = validate_clip(clip_audio, self.cfg)
        duration_s = len(clip_audio) / self.cfg.sample_rate
        severity = report.worst

        if severity == Severity.PASS:
            self._last_clip_info = (
                f"Last clip: [green]{duration_s:.1f}s, {report.rms_dbfs:.0f} dBFS RMS, GOOD[/green]"
            )
        elif severity == Severity.WARN:
            warns = ", ".join(r.check for r in report.results if r.severity == Severity.WARN)
            self._last_clip_info = (
                f"Last clip: [yellow]{duration_s:.1f}s, {report.rms_dbfs:.0f} dBFS RMS, WARN: {warns}[/yellow]"
            )
        else:
            reasons = ", ".join(report.reject_reasons)
            self._last_clip_info = (
                f"Last clip: [red]{duration_s:.1f}s, {report.rms_dbfs:.0f} dBFS RMS, REJECTED: {reasons}[/red]"
            )

    def update_undo(self):
        self._holding = False
        if self._positive_count > 0:
            self._positive_count -= 1
        self._last_clip_info = "[yellow]Last clip undone[/yellow]"

    def refresh(self):
        """Refresh the live display."""
        if self._live is not None:
            self._live.update(self._build_display())

    def start_live(self) -> Live:
        """Create and return the Live context manager."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        return self._live

    def print_summary(self, positive: int, rejected: int,
                      explicit_neg: int, gap_neg: int, negative_duration: float):
        """Print final summary after recording."""
        self.console.print()
        self.console.print("[bold]Collection Summary[/bold]")
        self.console.print(f"  Positive samples saved: [green]{positive}[/green]")
        if rejected > 0:
            self.console.print(f"  Positive rejected:     [red]{rejected}[/red]")
        neg_m, neg_s = divmod(int(negative_duration), 60)
        self.console.print(f"  Explicit negatives:    [red]{explicit_neg}[/red]")
        self.console.print(f"  Gap negatives:         {gap_neg}")
        self.console.print(f"  Total negative audio:  {neg_m}m {neg_s:02d}s")
        self.console.print(f"  Output: {self.cfg.output_dir}")
        self.console.print()
