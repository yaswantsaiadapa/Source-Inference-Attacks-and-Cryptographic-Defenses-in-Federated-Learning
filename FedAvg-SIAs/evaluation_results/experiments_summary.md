# Master Experimental Results Summary

**Random Guessing Baseline**: 10.0% ($K = 10$ clients)

| Phase | Defense Mechanism | Configuration | Test Accuracy (%) | SIA Attack Accuracy (%) | Privacy Status |
|---|---|---|---|---|---|
| Phase 2: Baseline | `none` | α=0.1, Seed=1 | 73.08% | **48.10%** | Vulnerable |
| Phase 2: Baseline | `none` | α=0.1, Seed=2 | 66.75% | **57.83%** | Vulnerable |
| Phase 2: Baseline | `none` | α=1.0, Seed=1 | 95.24% | **20.30%** | Partially Protected |
| Phase 2: Baseline | `none` | α=1.0, Seed=2 | 96.09% | **19.40%** | Fully Protected (~Random Guess) |
| Phase 2: Baseline | `none` | α=10.0, Seed=1 | 95.64% | **14.10%** | Fully Protected (~Random Guess) |
| Phase 2: Baseline | `none` | α=10.0, Seed=2 | 95.98% | **12.80%** | Fully Protected (~Random Guess) |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=1, σ=0.001 | 70.80% | **43.50%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=2, σ=0.001 | 69.02% | **58.26%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=1, σ=0.01 | 70.83% | **43.10%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=2, σ=0.01 | 67.92% | **57.39%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=1, σ=0.05 | 67.62% | **45.90%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=2, σ=0.05 | 65.48% | **55.65%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=1, σ=0.1 | 39.55% | **44.00%** | Vulnerable |
| Phase 3: DP Defense | `dp` | α=0.1, Seed=2, σ=0.1 | 27.13% | **43.04%** | Vulnerable |
| Phase 4: SecAgg Defense | `secagg` | α=0.1, Seed=1 | 61.47% | **8.50%** | Fully Protected (~Random Guess) |
| Phase 4: SecAgg Defense | `secagg` | α=0.1, Seed=2 | 68.75% | **4.78%** | Fully Protected (~Random Guess) |
| Phase 4: DP + SecAgg | `dp_secagg` | α=0.1, Seed=1, σ=0.01 | 60.69% | **8.50%** | Fully Protected (~Random Guess) |
| Phase 4: DP + SecAgg | `dp_secagg` | α=0.1, Seed=2, σ=0.01 | 68.98% | **4.78%** | Fully Protected (~Random Guess) |