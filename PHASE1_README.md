# Phase 1: Serial Baseline - Pearson Correlation Recommendation System

## Overview
This is the serial baseline implementation for the Parallel Pearson Correlation Based Recommendation Systems project.

## Files
- `main.cpp` - Main program demonstrating the serial baseline
- `recommendation.h` - Header file with class definition
- `recommendation.cpp` - Implementation of recommendation algorithm
- `Makefile` - Build script for Linux/WSL
- `CMakeLists.txt` - CMake configuration for cross-platform builds

## Compilation

### Option 1: Using CMake (Recommended)
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Option 2: Using Makefile (Linux/WSL)
```bash
make
make run
```

### Option 3: Direct Compilation with g++
```bash
g++ -std=c++17 -O2 -o phase1_serial main.cpp recommendation.cpp
```

### Option 4: Direct Compilation with MSVC (Windows)
```cmd
cl /O2 /std:c++latest main.cpp recommendation.cpp
```

## Features Implemented

### 1. Data Loading
- `generateRandomRatings()` - Generate sparse user-item matrix
- `loadRatingsFromCSV()` - Load ratings from CSV file

### 2. Pearson Correlation
- `calculatePearsonCorrelation()` - Calculate correlation between two users
- Handles sparse ratings and common items only

### 3. Collaborative Filtering
- `findNearestNeighbors()` - Find k-nearest neighbors based on correlation
- `predictRating()` - Predict rating for unrated item using weighted average of similar users

### 4. Accuracy Evaluation
- `validateAccuracy()` - Validate using 20% held-out test set
- `calculateRMSE()` - Compute Root Mean Square Error

## Algorithm Details

### Pearson Correlation Formula
$$r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n}(Y_i - \bar{Y})^2}}$$

### Rating Prediction
$$\hat{r}_{u,i} = \frac{\sum_{s \in S} w(u,s) \cdot r_{s,i}}{\sum_{s \in S} |w(u,s)|}$$

Where:
- $r_{u,i}$ = predicted rating for user u on item i
- $w(u,s)$ = Pearson correlation weight between users u and s
- $r_{s,i}$ = rating of similar user s on item i
- $S$ = set of k-nearest neighbors who rated item i

### RMSE
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

## Running the Program

```bash
./bin/phase1_serial
```

### Output
The program produces:
1. Dataset configuration summary
2. Sample ratings matrix
3. Pearson correlations between users
4. Top k-nearest neighbors
5. Single item prediction example
6. All items predictions for a user
7. RMSE accuracy on validation set
8. Performance timing information

## Parameters (Adjustable in main.cpp)

```cpp
int num_users = 100;      // Number of users
int num_items = 500;      // Number of items
int k_neighbors = 10;     // Number of neighbors for prediction
double sparsity = 0.8;    // Percentage of missing ratings (0-1)
```

## Expected Performance (Serial Baseline)

- **Dataset Size**: 100 users × 500 items (with 80% sparsity)
- **Algorithm Complexity**: O(users² × items)
- **Runtime**: ~1-5 seconds (depending on hardware)
- **RMSE**: ~0.5-1.0 (depends on dataset and parameters)

## Next Steps for Parallelization

Phase 2 will parallelize:
1. Correlation computation (embarrassingly parallel)
2. Neighbor search (parallel reduction)
3. Rating predictions (batch processing)
4. Validation (data decomposition)

### Parallelization Strategy:
- **OpenMP**: Thread-level parallelism for shared memory
- **Pthreads**: Fine-grained thread control
- **MPI**: Process-level parallelism for distributed memory
- **CUDA**: GPU acceleration for matrix operations
- **Hybrid**: Combine multiple approaches

## References

[1] Breese, J. S., Heckerman, D., & Kadie, C. (2013). "Empirical analysis of predictive algorithms for collaborative filtering." arXiv preprint arXiv:1301.7363.

[2] Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). "Item-based collaborative filtering recommendation algorithms." In Proceedings of the 10th international conference on World Wide Web (pp. 285–295).
