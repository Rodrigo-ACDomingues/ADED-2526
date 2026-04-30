import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# COLORS
# ----------------------------
WORKER_COLORS = {
    4:  ["#b30000", "#e34a33", "#fc8d59"],
    8:  ["#08519c", "#3182bd", "#6baed6"],
    12: ["#006d2c", "#31a354", "#74c476"]
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
# 2. Split blocks correctly (S/M/L from file)
# ----------------------------
def split_blocks_with_size(text):
    lines = text.splitlines()

    current_size = None
    blocks = []
    current_block = []

    for line in lines:
        line = line.strip()

        # detect size markers
        if line in ["S", "M", "L"]:
            current_size = line
            continue

        # new block start
        if "prompt eval time" in line:
            if current_block:
                blocks.append((current_size, "\n".join(current_block)))
                current_block = []

        current_block.append(line)

    if current_block:
        blocks.append((current_size, "\n".join(current_block)))

    return blocks


# ----------------------------
# 3. Parse block
# ----------------------------
def parse_block(text, meta, prompt_size):
    prompt = re.search(
        r'^prompt eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens.*?\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)',
        text,
        re.MULTILINE
    )

    eval_ = re.search(
        r'^\s*eval time =\s+([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens.*?\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)',
        text,
        re.MULTILINE
    )

    total = re.search(
        r'^total time =\s+([\d.]+)\s+ms',
        text,
        re.MULTILINE
    )

    if not (prompt and eval_ and total):
        return None

    return {
        **meta,
        "prompt_size": prompt_size,

        # tokens
        "prompt_tokens": int(prompt.group(2)),
        "gen_tokens": int(eval_.group(2)),

        # latência
        "ttft_ms": float(prompt.group(1)),
        "tpot_ms": float(eval_.group(3)),

        # throughput CORRETO
        "throughput_prefill": float(prompt.group(4)),
        "throughput_decode": float(eval_.group(4)),

        "total_time_ms": float(total.group(1))
    }

# ----------------------------
# 4. Load dataset
# ----------------------------
def load_all_data(folder):
    rows = []

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue

        try:
            meta = parse_filename(fname)
        except ValueError:
            print(f"[WARN] bad filename: {fname}")
            continue

        path = os.path.join(folder, fname)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_size = None
        buffer = []

        for line in lines:
            line = line.strip()

            # detect size marker
            if line in ["S", "M", "L"]:
                current_size = line
                continue

            buffer.append(line)

            # detect end of block
            if line.startswith("total time"):
                text = "\n".join(buffer)

                parsed = parse_block(text, meta, current_size)
                if parsed:
                    rows.append(parsed)

                buffer = []

    return pd.DataFrame(rows)

# ----------------------------
# 5. Debug check
# ----------------------------
def debug_ttft(df):
    print("\n=== TTFT sanity check ===")
    for _, r in df.iterrows():
        print(f"{r['prompt_size']} -> {r['ttft_ms']:.2f} ms")

def debug_ttft_order(df):
    print("\n=== TTFT ORDER DEBUG ===")

    stats = df.groupby("prompt_size")["ttft_ms"].agg(["count", "min", "mean", "max"])
    print(stats)

    # garantir ordem correta
    try:
        s = stats.loc["S", "mean"]
        m = stats.loc["M", "mean"]
        l = stats.loc["L", "mean"]

        print("\n=== MEAN CHECK ===")
        print(f"S mean: {s:.2f}")
        print(f"M mean: {m:.2f}")
        print(f"L mean: {l:.2f}")

        if s < m < l:
            print("✅ OK: S < M < L (monotonic)")
        else:
            print("❌ BROKEN ORDER: expected S < M < L")

    except KeyError as e:
        print(f"❌ Missing class: {e}")

    # checks por sample (mais forte)
    print("\n=== SAMPLE ORDER CHECK ===")

    for i, row in df.iterrows():
        print(f"{row['prompt_size']} -> {row['ttft_ms']:.2f} ms")

# ----------------------------
# 6. Plots
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

    for size in ["S", "M", "L"]:
        subset = df[df["prompt_size"] == size]

        plt.scatter(
            subset["total_tokens"],
            subset["total_time_ms"],
            label=size,
            alpha=0.7
        )

    plt.xlabel("Total tokens")
    plt.ylabel("Total time (ms)")
    plt.title("Total Time vs Tokens (S/M/L)")
    plt.legend(title="Prompt Size")
    plt.grid(True)

    plt.savefig(os.path.join(outdir, "total_time_vs_tokens.png"))
    plt.clf()

def plot_tpot_vs_threads(df, outdir):
    grouped = df.groupby(
        ["workers", "threads", "prompt_size"]
    ).mean(numeric_only=True).reset_index()

    sizes = ["S", "M", "L"]

    for w in sorted(grouped["workers"].unique()):
        subset_w = grouped[grouped["workers"] == w]

        for i, size in enumerate(sizes):
            s = subset_w[subset_w["prompt_size"] == size]

            if s.empty:
                continue

            color = WORKER_COLORS[w][i]

            plt.plot(
                s["threads"],
                s["tpot_ms"],
                marker='o',
                color=color,
                label=f"W={w} | {size}"
            )

    plt.xlabel("Threads")
    plt.ylabel("TPOT (ms/token)")
    plt.title("TPOT vs Threads")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "tpot_vs_threads.png"),
        bbox_inches="tight"
    )
    plt.clf()

def plot_tpot_vs_prompt(df, outdir):
    import os
    import matplotlib.pyplot as plt

    size_map = {"S": 0, "M": 1, "L": 2}

    x = df["prompt_size"].map(size_map)

    plt.scatter(
        x,
        df["tpot_ms"],
        alpha=0.7
    )

    plt.xticks([0, 1, 2], ["S", "M", "L"])

    plt.xlabel("Prompt Size")
    plt.ylabel("TPOT (ms/token)")
    plt.title("TPOT vs Prompt Size")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, "tpot_vs_prompt.png"),
        bbox_inches="tight"
    )

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
# 7. Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] Loading data...")
    df = load_all_data(args.input_dir)

    if df.empty:
        print("[ERROR] no data parsed")
        return

    df["prompt_size"] = pd.Categorical(
        df["prompt_size"],
        categories=["S", "M", "L"],
        ordered=True
    )

    print(f"[INFO] Loaded {len(df)} samples")

    # debug
    #debug_ttft(df)

    #debug_ttft_order(df)

    # save
    df.to_csv(os.path.join(args.outdir, "parsed_data.csv"), index=False)

    # Plots
    print("[INFO] Generating plots...")
    plot_decode_throughput_vs_threads(df, args.outdir)
    plot_prefill_throughput_vs_threads(df, args.outdir)
    plot_prefill_decode_ratio_vs_threads(df, args.outdir)
    plot_ttft_vs_threads(df, args.outdir)
    plot_ttft_vs_prompt(df, args.outdir)
    plot_total_time_vs_tokens(df, args.outdir)
    plot_tpot_vs_threads(df, args.outdir)
    plot_tpot_vs_prompt(df, args.outdir)

    # Summary table (com S/M/L)
    build_summary_table(df, args.outdir)

    print("[INFO] Done")


if __name__ == "__main__":                                                                        
    main()