import argparse, subprocess, sys, shlex, os

def main():
    ap = argparse.ArgumentParser(description="TSV → TSV wrapper for the onto-annotate CLI")
    ap.add_argument("--input", required=True, help="TSV input path")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--outdir", required=True, help="Output directory (TSV goes here)")
    ap.add_argument("--refresh", action="store_true", help="Force refresh ontology cache")
    ap.add_argument("--no_openai", action="store_true", help="Disable OpenAI fallback")
    ap.add_argument("--verbose", action="count", default=0, help="-v or -vv for more logs")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "onto_annotate.cli",
        "annotate",
        "--config", args.config,
        "--input_file", args.input,
        "--output_dir", args.outdir,
    ]
    if args.refresh:   cmd.append("--refresh")
    if args.no_openai: cmd.append("--no_openai")
    for _ in range(args.verbose): cmd.append("-v")

    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    env = os.environ.copy()
    # Ensure OPENAI_API_KEY is set in env if you plan to use OpenAI mode
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[run_pipeline] CLI failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
