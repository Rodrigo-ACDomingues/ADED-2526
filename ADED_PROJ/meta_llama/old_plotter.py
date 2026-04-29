import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

""" WORKER_COLORS = {
    4:  ["#d73027", "#4575b4", "#1a9850"],   # fortes (vermelho/azul/verde)
    8:  ["#fc8d59", "#91bfdb", "#a6d96a"],   # intermédios (laranja/azul/verde suave)
    12: ["#fee08b", "#8ecae6", "#e5f5e0"]    # claros (amarelo/azul claro/verde claro)
} """

WORKER_COLORS = {
    4:  ["#b30000", "#e34a33", "#fc8d59"],   # quentes fortes (vermelho dominante)
    8:  ["#08519c", "#3182bd", "#6baed6"],   # frias fortes (azul saturado)
    12: ["#006d2c", "#31a354", "#74c476"]    # verdes mais vivos
}

# ----------------------------
# 1. Filename parsing
# ----------------------------
def parse_filename(filename):
    match = re.search(r'W(\d+)T(\d+)', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    
    return {
        "workers": int(match.group(1)),
        "threads": int(match.group(2))
    }


# ----------------------------
# 2. Split blocks WITH prompt size
# ----------------------------
def split_blocks_with_size(text):
    lines = text.splitlines()
    
    current_size = None
    blocks = []
    current_block = []

    for line in lines:
        line = line.strip()

        # Detect S / M / L headers
        if line in ["S", "M", "L"]:
            current_size = line
            continue

        # Start new block
        if "prompt eval time" in line:
            if current_block:
                blocks.append((current_size, "\n".join(current_block)))
                current_block = []

        current_block.append(line)

    if current_block:
        blocks.append((current_size, "\n".join(current_block)))

    return blocks


# ----------------------------
# 3. Parse single block
# ----------------------------
def parse_block(text, meta, prompt_size):
    prompt = re.search(
        r'^\s*prompt eval time =\s+([\d.]+) ms / \s*(\d+) tokens.*?\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)',
        text,
        re.MULTILINE
    )
    
    eval_ = re.search(
        r'^\s*eval time =\s+([\d.]+) ms / \s*(\d+) tokens.*?\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)',
        text,
        re.MULTILINE
    )
    
    total = re.search(
        r'total time =\s+([\d.]+) ms / \s*(\d+) tokens',
        text
    )
    
    if not (prompt and eval_ and total):
        return None
    
    return {
        **meta,
        "prompt_size": prompt_size,
        "prompt_tokens": int(prompt.group(2)),
        "gen_tokens": int(eval_.group(2)),
        "ttft_ms": float(prompt.group(1)),
        "tpot_ms": float(eval_.group(3)),
        "throughput_prefill": float(prompt.group(4)),
        "throughput_decode": float(eval_.group(4)),
        "total_time_ms": float(total.group(1))
    }


# ----------------------------
# 4. Load all files
# ----------------------------
def load_all_data(folder):
    rows = []
    
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        
        try:
            meta = parse_filename(fname)
        except ValueError:
            print(f"[WARN] Skipping file with invalid name: {fname}")
            continue
        
        path = os.path.join(folder, fname)
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        blocks = split_blocks_with_size(content)
        
        for size, b in blocks:
            parsed = parse_block(b, meta, size)
            if parsed:
                rows.append(parsed)
    
    return pd.DataFrame(rows)


# ----------------------------
# 5. Plotting
# ----------------------------
def plot_decode_throughput_vs_threads(df, outdir):
    for w in sorted(df["workers"].unique()):
        subset = df[df["workers"] == w]
        
        grouped = subset.groupby(
            ["threads", "prompt_size"]
        ).mean(numeric_only=True).reset_index()

        for i, size in enumerate(["S", "M", "L"]):
            s = grouped[grouped["prompt_size"] == size]

            color = WORKER_COLORS[w][i]

            plt.plot(
                s["threads"],
                s["throughput_decode"],
                marker='o',
                color=color,
                label=f"W={w} | {size}"
            )
    
    plt.xlabel("Threads")
    plt.ylabel("Decode Throughput (tokens/s)")
    plt.title("Decode Throughput vs Threads")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "decode_throughput_vs_threads.png"), bbox_inches="tight")
    plt.clf()

def plot_prefill_throughput_vs_threads(df, outdir):
    for w in sorted(df["workers"].unique()):
        subset = df[df["workers"] == w]
        
        grouped = subset.groupby(
            ["threads", "prompt_size"]
        ).mean(numeric_only=True).reset_index()

        for i, size in enumerate(["S", "M", "L"]):
            s = grouped[grouped["prompt_size"] == size]

            color = WORKER_COLORS[w][i]

            plt.plot(
                s["threads"],
                s["throughput_prefill"],
                marker='o',
                color=color,
                label=f"W={w} | {size}"
            )
    
    plt.xlabel("Threads")
    plt.ylabel("Prefill Throughput (tokens/s)")
    plt.title("Prefill Throughput vs Threads")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "prefill_throughput_vs_threads.png"), bbox_inches="tight")
    plt.clf()

def plot_prefill_decode_ratio_vs_threads(df, outdir):
    grouped = df.groupby(
        ["workers", "threads", "prompt_size"]
    ).mean(numeric_only=True).reset_index()
    
    grouped["prefill_decode_ratio"] = (
        grouped["throughput_prefill"] /
        grouped["throughput_decode"]
    )
    
    for w in sorted(grouped["workers"].unique()):
        subset_w = grouped[grouped["workers"] == w]
        
        for i, size in enumerate(["S", "M", "L"]):
            s = subset_w[subset_w["prompt_size"] == size]

            color = WORKER_COLORS[w][i]

            plt.plot(
                s["threads"],
                s["prefill_decode_ratio"],
                marker='o',
                color=color,
                label=f"W={w} | {size}"
            )
    
    plt.xlabel("Threads")
    plt.ylabel("Prefill / Decode Throughput Ratio")
    plt.title("Prefill vs Decode Efficiency Ratio vs Threads")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "prefill_decode_ratio_vs_threads.png"), bbox_inches="tight")
    plt.clf()

def plot_ttft_vs_threads(df, outdir):
    import os
    import matplotlib.pyplot as plt

    grouped = df.groupby(
        ["workers", "threads", "prompt_size"]
    ).mean(numeric_only=True).reset_index()

    sizes = ["S", "M", "L"]

    for size in sizes:
        plt.figure(figsize=(7, 5))

        subset_size = grouped[grouped["prompt_size"] == size]

        for w in sorted(subset_size["workers"].unique()):
            s = subset_size[subset_size["workers"] == w]

            color = WORKER_COLORS[w][sizes.index(size)]

            plt.plot(
                s["threads"],
                s["ttft_ms"],
                marker='o',
                color=color,
                label=f"W={w}"
            )

        plt.xlabel("Threads")
        plt.ylabel("TTFT (ms)")
        plt.title(f"TTFT vs Threads ({size})")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(outdir, f"ttft_vs_threads_{size}.png"),
            bbox_inches="tight"
        )

        plt.clf()

def plot_ttft_vs_prompt(df, outdir):
    import os
    import matplotlib.pyplot as plt

    size_map = {"S": 0, "M": 1, "L": 2}

    x = df["prompt_size"].map(size_map)

    plt.scatter(
        x,
        df["ttft_ms"]
    )

    plt.xticks([0, 1, 2], ["S", "M", "L"])

    plt.xlabel("Prompt Size")
    plt.ylabel("TTFT (ms)")
    plt.title("TTFT vs Prompt Size")

    plt.grid(True)

    plt.savefig(
        os.path.join(outdir, "ttft_vs_prompt.png"),
        bbox_inches="tight"
    )

    plt.clf()

def plot_total_time_vs_tokens(df, outdir):
    df["total_tokens"] = df["prompt_tokens"] + df["gen_tokens"]
    
    plt.scatter(df["total_tokens"], df["total_time_ms"])
    
    plt.xlabel("Total tokens")
    plt.ylabel("Total time (ms)")
    plt.title("Total Time vs Tokens")
    plt.grid(True)
    
    plt.savefig(os.path.join(outdir, "total_time_vs_tokens.png"))
    plt.clf()


# ----------------------------
# 6. Summary Table (WITH S/M/L)
# ----------------------------
def build_summary_table(df, outdir):
    table = df.groupby(["workers", "threads", "prompt_size"]).agg({
        "ttft_ms": "mean",
        "tpot_ms": "mean",
        "throughput_prefill": "mean",
        "throughput_decode": "mean",
        "total_time_ms": "mean"
    }).reset_index()

    table = table.rename(columns={
        "ttft_ms": "TTFT (ms)",
        "tpot_ms": "TPOT (ms/token)",
        "throughput_prefill": "Prefill throughput (tok/s)",
        "throughput_decode": "Decode throughput (tok/s)",
        "total_time_ms": "Total time (ms)"
    })

    csv_path = os.path.join(outdir, "summary_table_by_prompt_size.csv")
    table.to_csv(csv_path, index=False)

    print("\n=== Summary Table (by prompt size) ===")
    try:
        print(table.to_markdown(index=False))
    except:
        print(table)

    print(f"\n[INFO] Saved table: {csv_path}")


# ----------------------------
# 7. Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze LLM benchmark logs")
    parser.add_argument("input_dir", help="Directory with .txt log files")
    parser.add_argument("--outdir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("[INFO] Loading data...")
    df = load_all_data(args.input_dir)
    
    if df.empty:
        print("[ERROR] No valid data found.")
        return
    
    df["prompt_size"] = pd.Categorical(
        df["prompt_size"],
        categories=["S", "M", "L"],
        ordered=True
    )
    
    print(f"[INFO] Loaded {len(df)} samples")
    
    # Save raw data
    csv_path = os.path.join(args.outdir, "parsed_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV: {csv_path}")
    
    # Plots
    print("[INFO] Generating plots...")
    plot_decode_throughput_vs_threads(df, args.outdir)
    plot_prefill_throughput_vs_threads(df, args.outdir)
    plot_prefill_decode_ratio_vs_threads(df, args.outdir)
    plot_ttft_vs_threads(df, args.outdir)
    plot_ttft_vs_prompt(df, args.outdir)
    plot_total_time_vs_tokens(df, args.outdir)

    # Summary table (com S/M/L)
    build_summary_table(df, args.outdir)
    
    print(f"[INFO] Outputs saved in: {args.outdir}")


if __name__ == "__main__":
    main()