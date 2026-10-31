"""Run every experiment of the paper in order.

    python experiments/run_all.py            # full settings (about 2 hours on 2 cores)
    python experiments/run_all.py --quick    # small settings (about 10 minutes)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = [
    "individual_perturbations.py",     # Table S3, Figure 4
    "factorial_interaction.py",        # Table 4, Table S2, Figure 5
    "stability_boundaries.py",         # Table S1, Figure 3, scheme edges
    "practical_retractions.py",        # Table 3 (uses scheme edges)
    "practical_retractions_real.py",   # Table S7 (real data)
    "robustness_30seeds.py",           # Table S6, 30-run divergence test
    "ablation.py",                     # Table 6
    "baselines.py",                    # Table 5
    "figS1_digits.py",                 # Figure S1
    "large_problem.py",                # spiked n = 10^4
    "replication_and_stability.py",    # Table S4, Section 3.2 checks
    "formats.py",                      # Table S5
    "convergence_floors.py",           # Figure 2
    "cost.py",                         # computational cost (Section 3.3)
    "paper_numbers.py",                # PAPER_NUMBERS.md (numbers quoted in the text)
    "compare_with_paper.py",           # COMPARISON.md
]

if __name__ == "__main__":
    args = sys.argv[1:]
    for script in ORDER:
        t0 = time.time()
        print(f"\n=== {script} ===", flush=True)
        extra = [] if script in ("compare_with_paper.py", "paper_numbers.py") else args
        subprocess.run([sys.executable, os.path.join(HERE, script), *extra], check=True)
        print(f"--- {script}: {time.time() - t0:.0f} s", flush=True)
