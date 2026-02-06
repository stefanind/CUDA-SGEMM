from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("./benchmark/kernel_results/data/kernel_results.csv")
OUT_DIR  = Path("./benchmark/kernel_results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(CSV_PATH)

    # basic cleanup
    df["kernel"] = df["kernel"].astype(str).str.strip()
    df["gpu"] = df["gpu"].astype(str).str.strip()
    for c in ["M","N","K","alpha","beta","ms","gflops","pct_of_cublas","max_abs","max_rel"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    shape = df[["M","N","K"]].value_counts().idxmax()
    sub = df[(df["M"]==shape[0]) & (df["N"]==shape[1]) & (df["K"]==shape[2])].copy()

    gpu = sub["gpu"].iloc[0]
    title_suffix = f"{gpu} | M=N=K={int(shape[0])}"

    # cuBLAS baseline
    cublas = sub[sub["kernel"].str.lower().str.contains("cublas")]
    cublas_gflops = float(cublas["gflops"].iloc[0]) if len(cublas) else None

    # --- Plot 1: GFLOPs by kernel (sorted) ---
    d = sub.sort_values("gflops", ascending=False)
    plt.figure()
    plt.bar(d["kernel"], d["gflops"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("GFLOPs")
    plt.title(f"SGEMM Throughput — {title_suffix}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gflops_by_kernel.png", dpi=220)
    plt.close()

    # --- Plot 2: Speedup vs cuBLAS ---
    if cublas_gflops and cublas_gflops > 0:
        d2 = sub.copy()
        d2["speedup_vs_cublas"] = d2["gflops"] / cublas_gflops
        d2 = d2.sort_values("speedup_vs_cublas", ascending=False)

        plt.figure()
        plt.bar(d2["kernel"], d2["speedup_vs_cublas"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Speedup vs cuBLAS")
        plt.axhline(1.0, linestyle="--")
        plt.title(f"Speedup vs cuBLAS — {title_suffix}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "speedup_vs_cublas.png", dpi=220)
        plt.close()

    # --- Plot 3: Runtime (ms) by kernel (sorted) ---
    d3 = sub.sort_values("ms", ascending=True)
    plt.figure()
    plt.bar(d3["kernel"], d3["ms"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Time (ms)")
    plt.title(f"SGEMM Runtime — {title_suffix}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ms_by_kernel.png", dpi=220)
    plt.close()

    # write top-kernels table for README
    top = sub.sort_values("gflops", ascending=False)[
        ["kernel","ms","gflops","pct_of_cublas"]
    ].head(10)

    (OUT_DIR / "top_kernels.md").write_text(top.to_markdown(index=False))
    print(f"Saved plots + table to: {OUT_DIR}")

if __name__ == "__main__":
    main()
