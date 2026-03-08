# EC7207: High Performance Computing – Group 11
## Pearson Correlation User-Based Collaborative Filtering

This project contains **serial** and **OpenMP parallel** implementations of a Pearson Correlation User-Based Collaborative Filtering recommender system.

---

## 📁 Project Structure

```
HPC/
├── serial_recommender.c        # Serial baseline implementation
├── openmp_recommender.c        # OpenMP parallel implementation
├── Makefile                    # Compilation script
└── README.md                   # This file
```

---

## 🔧 Compilation

### Compile both versions:
```bash
make all
```

### Compile individual versions:
```bash
make serial_rec       # Serial version only
make openmp_rec       # OpenMP version only
```

### Clean compiled binaries:
```bash
make clean
```

---

## ▶️ Running the Programs

### Serial Version (Baseline)
```bash
./serial_rec [num_users] [num_items]
```

**Examples:**
```bash
./serial_rec                    # Default: 1000 users, 1000 items
./serial_rec 500 500           # 500 users, 500 items
./serial_rec 2000 1500         # 2000 users, 1500 items
```

### OpenMP Version (Parallel)
```bash
OMP_NUM_THREADS=<N> ./openmp_rec [num_users] [num_items]
```

**Examples:**
```bash
OMP_NUM_THREADS=1 ./openmp_rec                    # Single thread
OMP_NUM_THREADS=4 ./openmp_rec 500 500           # 4 threads, 500×500
OMP_NUM_THREADS=8 ./openmp_rec 2000 1500        # 8 threads, 2000×1500
```

---

## 📊 Algorithm Overview

### Phases:
1. **Data Generation** – Sparse rating matrix with test set
2. **User Means** – Average rating per user
3. **Similarity Matrix** – Pearson correlation between users
4. **Predictions** – Top-K user-based collaborative filtering
5. **Evaluation** – MAE (Mean Absolute Error) on test set

### Key Parameters:
- **SPARSITY**: 0.70 (70% of entries unrated)
- **TOP_K**: 20 neighbors used for prediction
- **SEED**: 42 (reproducible RNG)
- **TEST_RATIO**: 0.10 (10% of known ratings held out)

---

## 🚀 Performance Notes

### Serial (`serial_recommender.c`):
- Uses `clock_gettime(CLOCK_MONOTONIC)`
- Single-threaded baseline

### OpenMP (`openmp_recommender.c`):
- Uses `omp_get_wtime()` for timing
- Parallel loops with strategic scheduling:
  - **User means**: `static` schedule
  - **Similarity matrix**: `dynamic(4)` schedule (load balancing)
  - **Predictions**: `dynamic(2)` schedule (nested parallelism)
  - **Evaluation**: `static` schedule with `reduction(+err)`

---

## 📈 Expected Output

Both programs output:
```
=== Pearson Correlation Recommender – [Serial/OpenMP] Version ===
    Users: N | Items: M | Top-K: 20 | Threads: T

[Data]   Users: N | Items: M | Sparsity: 70% | Test ratings: X
[Timing] Data generation    : T₁ s
[Timing] User mean compute  : T₂ s
[Timing] Similarity matrix  : T₃ s
[Check]  Sim-matrix checksum: ###.######
[Timing] Prediction phase   : T₄ s
[Eval]   MAE on test set    : ##.####  (test size: X)
[Timing] Total (sim+pred)   : T₃+T₄ s

--- Sample Predictions (first 5 users, 5 items) ---
User\Item  Item0  Item1  Item2  Item3  Item4
...
```

---

## 🔬 Testing

Run quick tests with:
```bash
make test_serial        # Run serial version (500×500)
make test_openmp        # Run OpenMP with 4 threads (500×500)
make test_all          # Run both
```

---

## ⚡ Performance Benchmarking

To benchmark different problem sizes and thread counts:

```bash
# Serial baseline
time ./serial_rec 1000 1000

# OpenMP with varying threads
for N in 1 2 4 8; do
    echo "=== $N threads ==="
    time OMP_NUM_THREADS=$N ./openmp_rec 1000 1000
done
```

---

## 📝 Code Features

- ✅ Dynamic memory management with proper cleanup
- ✅ Reproducible results (`SEED = 42`)
- ✅ Data sparsity simulation (70% unrated)
- ✅ Efficient flat array access (cache-friendly)
- ✅ Pearson correlation with mean-centered deviations
- ✅ Top-K neighbor filtering
- ✅ MAE evaluation metric
- ✅ Detailed timing breakdown
- ✅ Thread-safe OpenMP parallelization
- ✅ Sample output table for verification

---

## 📚 References

- Pearson Correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
- Collaborative Filtering: https://en.wikipedia.org/wiki/Collaborative_filtering
- OpenMP Documentation: https://www.openmp.org/

---

**Group 11 – EC7207: High Performance Computing**
