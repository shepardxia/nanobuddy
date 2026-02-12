"""Wake word sample collection — recording, segmentation, and validation."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

from .config import CollectConfig
from .dashboard import Dashboard
from .recorder import Recorder
from .segmenter import segment_recording
from .validator import Severity, validate_clip


def run_collect(
    phrase: str,
    target: int,
    output: str,
    device: int | None,
    auto: bool,
    auto_interval: float,
    auto_duration: float,
) -> None:
    """Record wake word samples interactively."""
    cfg = CollectConfig(
        phrase=phrase,
        target_positive=target,
        output_dir=output,
        input_device=device,
        auto_capture=auto,
        auto_interval_s=auto_interval,
        auto_duration_s=auto_duration,
    )
    if cfg.auto_capture:
        cfg.keypress_trim_start_ms = 0
        cfg.keypress_trim_end_ms = 0

    cfg.ensure_dirs()
    dash = Dashboard(cfg)

    existing_pos = len(list(cfg.positive_dir.glob(f"{cfg.phrase_slug}_*.wav")))
    existing_neg = len(list(cfg.negative_dir.glob("neg_*.wav")))
    if existing_pos or existing_neg:
        dash.console.print(f"[dim]Continuing: {existing_pos} positive, {existing_neg} negative clips exist.[/dim]")

    dash.print_banner()

    def on_hold_start(kind="pos"):
        dash.update_hold_start(kind)
        dash.refresh()

    def on_hold_end(hold, clip: np.ndarray):
        dash.update_hold_end(hold, clip)
        dash.refresh()

    def on_undo():
        dash.update_undo()
        dash.refresh()

    def on_tick(elapsed: float):
        dash.update_tick(elapsed)
        dash.refresh()

    def on_auto_flash(color):
        dash.set_flash(color)
        dash.refresh()

    recorder = Recorder(
        cfg,
        on_hold_start=on_hold_start,
        on_hold_end=on_hold_end,
        on_undo=on_undo,
        on_tick=on_tick,
        on_auto_flash=on_auto_flash if cfg.auto_capture else None,
    )

    with dash.start_live():
        session = recorder.record()

    dash.console.print()
    dash.console.print(f"[dim]Recording stopped. {len(session.holds)} marks in {session.duration_s:.1f}s.[/dim]")

    if not session.holds and session.duration_s < 1.0:
        dash.console.print("[yellow]No marks recorded. Nothing to save.[/yellow]")
        return

    dash.console.print("[dim]Segmenting clips...[/dim]")
    result = segment_recording(session.audio, session.holds, cfg)

    meta = {
        "phrase": cfg.phrase,
        "timestamp": datetime.now().isoformat(),
        "duration_s": session.duration_s,
        "sample_rate": session.sample_rate,
        "positive_saved": result.positive_saved,
        "positive_rejected": result.positive_rejected,
        "negative_saved": result.negative_saved,
        "holds": [{"start": h.start_sample, "end": h.end_sample} for h in session.holds],
    }

    sessions_path = Path(cfg.output_dir) / "sessions.jsonl"
    with open(sessions_path, "a") as f:
        f.write(json.dumps(meta) + "\n")

    dash.print_summary(
        result.positive_saved, result.positive_rejected,
        result.explicit_negative_saved, result.gap_negative_saved,
        result.negative_duration_s,
    )


def run_validate(path: str) -> None:
    """Validate quality of collected samples."""
    console = Console()
    cfg = CollectConfig()
    positive_dir = Path(path) / "positive"

    if not positive_dir.exists():
        console.print(f"[red]No positive/ directory found in {path}[/red]")
        return

    wavs = sorted(positive_dir.glob("*.wav"))
    if not wavs:
        console.print(f"[yellow]No .wav files found in {positive_dir}[/yellow]")
        return

    table = Table(title=f"Validation Report ({len(wavs)} clips)")
    table.add_column("File", style="cyan", max_width=30)
    table.add_column("Duration", justify="right")
    table.add_column("RMS dBFS", justify="right")
    table.add_column("Speech ms", justify="right")
    table.add_column("Status")

    pass_count = warn_count = reject_count = 0

    for wav_path in wavs:
        audio, sr = sf.read(wav_path, dtype="int16")
        report = validate_clip(audio, cfg)
        severity = report.worst

        if severity == Severity.PASS:
            status = "[green]PASS[/green]"
            pass_count += 1
        elif severity == Severity.WARN:
            warns = ", ".join(r.check for r in report.results if r.severity == Severity.WARN)
            status = f"[yellow]WARN: {warns}[/yellow]"
            warn_count += 1
        else:
            reasons = ", ".join(report.reject_reasons)
            status = f"[red]REJECT: {reasons}[/red]"
            reject_count += 1

        table.add_row(wav_path.name, f"{report.duration_s:.2f}s",
                      f"{report.rms_dbfs:.1f}", str(report.speech_ms), status)

    console.print(table)
    console.print()
    console.print(f"  [green]Pass: {pass_count}[/green]  "
                  f"[yellow]Warn: {warn_count}[/yellow]  "
                  f"[red]Reject: {reject_count}[/red]  "
                  f"Total: {len(wavs)}")


def run_stats(path: str) -> None:
    """Show statistics for a collected dataset."""
    console = Console()
    collect_path = Path(path)

    for subdir in ("positive", "negative", "rejected"):
        d = collect_path / subdir
        if not d.exists():
            console.print(f"  {subdir + ':':12s} [dim]not found[/dim]")
            continue

        wavs = list(d.glob("*.wav"))
        total_duration = 0.0
        for w in wavs:
            audio, sr = sf.read(w, dtype="int16")
            total_duration += len(audio) / sr
        minutes, seconds = divmod(int(total_duration), 60)
        console.print(f"  {subdir + ':':12s} {len(wavs):4d} clips  ({minutes}m {seconds:02d}s)")
