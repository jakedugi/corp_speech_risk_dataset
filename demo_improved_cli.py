#!/usr/bin/env python3
"""
Demo script showing the improved disso_prompt_cli.py handling malformed JSON automatically.
This demonstrates the full workflow with the actual output_3_1_1.json file.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the results."""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ SUCCESS!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    else:
        print(f"❌ FAILED with exit code {result.returncode}")
        if result.stderr:
            print("Error:")
            print(result.stderr)

    return result.returncode == 0


def main():
    print("IMPROVED DISSO_PROMPT_CLI DEMO")
    print("=" * 60)
    print("This demo shows how the CLI now handles malformed JSON automatically")
    print("using json5 for robust parsing and smart quote normalization.")

    # Path to the actual output file
    json_file = "src/corp_speech_risk_dataset/disso_prompt/output_3_1_1.json"
    csv_file = "src/corp_speech_risk_dataset/disso_prompt/disso_sections.csv"

    if not Path(json_file).exists():
        print(f"❌ File not found: {json_file}")
        sys.exit(1)

    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)

    # Demo 1: Paste the JSON file content (with automatic cleaning)
    success = run_command(
        [
            "uv",
            "run",
            "python",
            "src/corp_speech_risk_dataset/disso_prompt/disso_prompt_cli.py",
            "paste-final",
            "--csv",
            csv_file,
            "--id",
            "3.1.1",
            "--from-file",
            json_file,
            "--store-latex",
            "--mark-done",
        ],
        "Hydrate all fields from malformed JSON and generate LaTeX",
    )

    if not success:
        print("\n⚠️  The paste-final command failed. This might be due to:")
        print("   - The section ID 3.1.1 doesn't exist in the CSV")
        print("   - Other validation issues")
        print("\nLet's check what sections are available...")

        # List available sections
        run_command(
            [
                "uv",
                "run",
                "python",
                "src/corp_speech_risk_dataset/disso_prompt/disso_prompt_cli.py",
                "hud",
                "--csv",
                csv_file,
            ],
            "Show available sections",
        )

        return

    # Demo 2: Export the LaTeX version
    run_command(
        [
            "uv",
            "run",
            "python",
            "src/corp_speech_risk_dataset/disso_prompt/disso_prompt_cli.py",
            "export",
            "--csv",
            csv_file,
            "--id",
            "3.1.1",
            "--format",
            "latex",
            "--out",
            "section_3_1_1.tex",
        ],
        "Export the section as LaTeX",
    )

    # Check if the LaTeX file was created
    if Path("section_3_1_1.tex").exists():
        print(f"\n✅ LaTeX file created: section_3_1_1.tex")
        print(f"   Size: {Path('section_3_1_1.tex').stat().st_size} bytes")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nKey improvements:")
    print("1. ✅ Automatically removes code fences (```json ... ```)")
    print("2. ✅ Converts smart quotes (" " ' ') to ASCII quotes")
    print("3. ✅ Handles trailing commas and JSON5 features")
    print("4. ✅ Robust parsing with json5 library")
    print("5. ✅ All fields are hydrated correctly from structured JSON")
    print("6. ✅ LaTeX generation works seamlessly")


if __name__ == "__main__":
    main()
