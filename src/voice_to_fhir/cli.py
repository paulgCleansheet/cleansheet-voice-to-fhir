"""
Command Line Interface

CLI for voice-to-FHIR pipeline.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="voice-to-fhir",
    help="Edge-deployable clinical voice documentation pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Audio file to process"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    workflow: str = typer.Option("general", "--workflow", "-w", help="Clinical workflow"),
    indent: int = typer.Option(2, "--indent", help="JSON indent"),
) -> None:
    """Process an audio file to FHIR Bundle."""
    from voice_to_fhir import Pipeline, PipelineConfig

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load config
        if config:
            progress.add_task("Loading configuration...", total=None)
            pipeline = Pipeline.from_config(config)
        else:
            progress.add_task("Initializing pipeline...", total=None)
            pipeline = Pipeline(PipelineConfig())

        # Process file
        task = progress.add_task("Processing audio...", total=None)
        bundle = pipeline.process_file(input_file, workflow)
        progress.update(task, completed=True)

    # Output
    json_output = pipeline.to_json(bundle, indent)

    if output:
        output.write_text(json_output)
        console.print(f"[green]Output saved to: {output}[/green]")
    else:
        console.print(json_output)

    # Show summary
    entry_count = len(bundle.get("entry", []))
    console.print(f"\n[dim]Generated {entry_count} FHIR resources[/dim]")


@app.command()
def capture(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    workflow: str = typer.Option("general", "--workflow", "-w", help="Clinical workflow"),
    duration: float = typer.Option(30.0, "--duration", "-d", help="Max duration (seconds)"),
) -> None:
    """Capture audio from microphone and process to FHIR."""
    from voice_to_fhir import Pipeline, PipelineConfig

    console.print("[bold]Starting audio capture...[/bold]")
    console.print("[dim]Speak now. Recording will stop after silence is detected.[/dim]")

    # Load config
    if config:
        pipeline = Pipeline.from_config(config)
    else:
        pipeline = Pipeline(PipelineConfig())

    try:
        bundle = pipeline.capture_and_process(duration, workflow)
    except KeyboardInterrupt:
        console.print("\n[yellow]Capture cancelled[/yellow]")
        raise typer.Exit(0)

    # Output
    json_output = pipeline.to_json(bundle)

    if output:
        output.write_text(json_output)
        console.print(f"[green]Output saved to: {output}[/green]")
    else:
        console.print(json_output)


@app.command()
def transcript(
    text: str = typer.Argument(..., help="Transcript text to process"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    workflow: str = typer.Option("general", "--workflow", "-w", help="Clinical workflow"),
) -> None:
    """Process transcript text directly to FHIR (skip transcription)."""
    from voice_to_fhir import Pipeline, PipelineConfig

    pipeline = Pipeline(PipelineConfig())
    bundle = pipeline.process_transcript(text, workflow)

    json_output = pipeline.to_json(bundle)

    if output:
        output.write_text(json_output)
        console.print(f"[green]Output saved to: {output}[/green]")
    else:
        console.print(json_output)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing audio files"),
    output_dir: Path = typer.Argument(..., help="Output directory for FHIR files"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    workflow: str = typer.Option("general", "--workflow", "-w", help="Clinical workflow"),
    pattern: str = typer.Option("*.wav", "--pattern", "-p", help="File pattern"),
) -> None:
    """Batch process audio files to FHIR."""
    from voice_to_fhir import Pipeline, PipelineConfig

    if not input_dir.exists():
        console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    files = list(input_dir.glob(pattern))
    if not files:
        console.print(f"[yellow]No files matching '{pattern}' in {input_dir}[/yellow]")
        raise typer.Exit(0)

    # Load pipeline
    if config:
        pipeline = Pipeline.from_config(config)
    else:
        pipeline = Pipeline(PipelineConfig())

    console.print(f"Processing {len(files)} files...")

    success_count = 0
    error_count = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(files))

        for audio_file in files:
            try:
                bundle = pipeline.process_file(audio_file, workflow)
                output_file = output_dir / f"{audio_file.stem}.json"
                pipeline.save(bundle, output_file)
                success_count += 1
            except Exception as e:
                console.print(f"[red]Error processing {audio_file}: {e}[/red]")
                error_count += 1

            progress.update(task, advance=1)

    console.print(f"\n[green]Processed: {success_count}[/green]")
    if error_count:
        console.print(f"[red]Errors: {error_count}[/red]")


@app.command()
def devices() -> None:
    """List available audio input devices."""
    from voice_to_fhir.capture import AudioCapture

    devices = AudioCapture.list_devices()

    if not devices:
        console.print("[yellow]No audio input devices found[/yellow]")
        raise typer.Exit(0)

    console.print("[bold]Available audio input devices:[/bold]\n")
    for device in devices:
        console.print(
            f"  [{device['index']}] {device['name']}"
            f"\n      Channels: {device['channels']}, "
            f"Sample Rate: {device['sample_rate']} Hz"
        )


@app.command()
def version() -> None:
    """Show version information."""
    from voice_to_fhir import __version__

    console.print(f"voice-to-fhir version {__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
