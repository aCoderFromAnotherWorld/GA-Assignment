"""
Generates a FULL per-generation audit trail of the GA run: one table (CSV) and
one plot (PNG) for every single generation, plus consolidated CSV/Excel files
and an animated GIF showing the population converging over time.

This is meant for presenting to an instructor who wants to see every step of
every iteration, not just the final answer.

Run with:
    cd src
    python3 generate_detailed_report.py
"""

import os
import shutil

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ackley import BOUNDS, ackley, ackley_vec
from genetic_algorithm import GAConfig, GeneticAlgorithm

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "detailed")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def reset_output_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def run_ga(seed: int = 42):
    print("Running GA with full per-generation logging enabled...")
    cfg = GAConfig(seed=seed)
    ga = GeneticAlgorithm(fitness_fn=ackley_vec, config=cfg)
    result = ga.run(verbose=True, log_details=True)
    print(
        f"Done: {result.generations_run} generations "
        f"({'converged' if result.converged else 'reached max generations'}), "
        f"best fitness = {result.best_fitness:.8f}"
    )
    return result, cfg


def save_per_generation_tables(pop_df: pd.DataFrame):
    print("Writing one CSV table per generation to output/detailed/tables/ ...")
    for gen, gen_df in pop_df.groupby("generation"):
        path = os.path.join(TABLES_DIR, f"gen_{gen:03d}.csv")
        gen_df.drop(columns="generation").to_csv(path, index=False)


def _ackley_background(resolution: int = 150):
    low, high = BOUNDS
    grid = np.linspace(low, high, resolution)
    X1, X2 = np.meshgrid(grid, grid)
    Z = ackley(X1, X2)
    return X1, X2, Z


def save_per_generation_plots(pop_df: pd.DataFrame, summary_df: pd.DataFrame):
    print("Rendering one plot per generation to output/detailed/plots/ ...")
    X1, X2, Z = _ackley_background()
    generations = sorted(pop_df["generation"].unique())
    _ = summary_df.set_index("generation") if not summary_df.empty else None

    frame_paths = []
    for gen in generations:
        gdf = pop_df[pop_df["generation"] == gen]
        elite = gdf[gdf["is_elite"]]
        rest = gdf[~gdf["is_elite"]]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        contour = ax.contourf(X1, X2, Z, levels=40, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="f(x1, x2)")
        ax.plot(
            0,
            0,
            marker="*",
            color="white",
            markersize=14,
            linestyle="None",
            label="True global minimum",
        )
        ax.scatter(
            rest["x1"],
            rest["x2"],
            s=22,
            color="red",
            alpha=0.8,
            edgecolors="black",
            linewidths=0.3,
            label="Population",
        )
        if len(elite):
            ax.scatter(
                elite["x1"],
                elite["x2"],
                s=70,
                color="orange",
                marker="D",
                edgecolors="black",
                linewidths=0.6,
                label="Elite (best)",
            )

        best_f = gdf["fitness"].min()
        title = f"Generation {gen} — best f = {best_f:.5f}"
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(BOUNDS)
        ax.set_ylim(BOUNDS)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"gen_{gen:03d}.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        frame_paths.append(path)

    return frame_paths


def save_animation(frame_paths, path, fps: int = 6):
    print(f"Building animated GIF ({len(frame_paths)} frames) -> {path}")
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(path, frames, fps=fps)


def save_excel_workbook(summary_df, pop_df, events_df, path):
    print(f"Writing consolidated Excel workbook -> {path}")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary (per generation)", index=False)
        pop_df.to_excel(writer, sheet_name="Population (all generations)", index=False)
        events_df.to_excel(writer, sheet_name="Events (all generations)", index=False)


def main():
    reset_output_dirs()
    result, _ = run_ga()

    pop_df = pd.DataFrame(result.population_log)
    events_df = pd.DataFrame(result.events_log)
    summary_df = pd.DataFrame(result.summary_log)

    # Consolidated long-format CSVs (easy to filter/pivot per generation in Excel)
    pop_csv = os.path.join(OUTPUT_DIR, "population_all_generations.csv")
    events_csv = os.path.join(OUTPUT_DIR, "events_all_generations.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "summary_all_generations.csv")
    pop_df.to_csv(pop_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {pop_csv}\nSaved: {events_csv}\nSaved: {summary_csv}")

    # One CSV table per generation (population only — the "table for each iteration")
    save_per_generation_tables(pop_df)

    # One plot per generation (population scattered over the Ackley surface)
    frame_paths = save_per_generation_plots(pop_df, summary_df)

    # Animated GIF stitched from all per-generation plots
    gif_path = os.path.join(OUTPUT_DIR, "evolution.gif")
    save_animation(frame_paths, gif_path)

    # One Excel workbook with everything, filterable by generation
    excel_path = os.path.join(OUTPUT_DIR, "GA_Detailed_Report.xlsx")
    save_excel_workbook(summary_df, pop_df, events_df, excel_path)

    print("\nAll detailed, per-generation outputs are in: src/output/detailed/")
    print(
        f"  - tables/gen_XXX.csv        ({len(pop_df['generation'].unique())} files, one per generation)"
    )
    print(
        f"  - plots/gen_XXX.png         ({len(frame_paths)} files, one per generation)"
    )
    print("  - evolution.gif             (all generations animated)")
    print(
        "  - GA_Detailed_Report.xlsx   (Summary / Population / Events sheets, filterable)"
    )
    print(
        "  - population_all_generations.csv / events_all_generations.csv / summary_all_generations.csv"
    )


if __name__ == "__main__":
    main()
