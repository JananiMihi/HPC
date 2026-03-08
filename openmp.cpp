#include "recommendation_openmp.h"
#include <chrono>
#include <iostream>
#include <iomanip>

void printHeader(const std::string& title) {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::endl;
}

int main() {
    printHeader("Phase 2a: OpenMP Parallelization - Pearson Correlation Recommendation System");

    // Parameters
    int num_users = 100;
    int num_items = 500;
    int k_neighbors = 10;
    double sparsity = 0.8;

    std::cout << "Dataset Configuration:" << std::endl;
    std::cout << "  Users: " << num_users << std::endl;
    std::cout << "  Items: " << num_items << std::endl;
    std::cout << "  Sparsity: " << (sparsity * 100) << "%" << std::endl;
    std::cout << "  K-neighbors: " << k_neighbors << std::endl;
    std::cout << std::endl;

    // Test with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    std::cout << "Operating System Information:" << std::endl;
    std::cout << "  Max threads available: " << omp_get_num_procs() << std::endl;
    std::cout << std::endl;

    for (int num_threads : thread_counts) {
        if (num_threads > omp_get_num_procs()) {
            std::cout << "Skipping " << num_threads << " threads (not available)" << std::endl;
            continue;
        }

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Running with " << num_threads << " thread(s)" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        RecommendationSystemOpenMP recommender(num_users, num_items, num_threads);

        // Generate random ratings
        std::cout << "Generating random ratings..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        recommender.generateRandomRatings(sparsity);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;

        // Display sample (only for 1 thread to avoid redundant output)
        if (num_threads == 1) {
            std::cout << "Sample of Ratings Matrix:" << std::endl;
            recommender.printRatings();
            std::cout << std::endl;
        }

        // Test 1: Calculate all correlations (PARALLEL)
        std::cout << "Test 1: Calculate All Correlations (Parallel)" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        auto correlations = recommender.calculateAllCorrelations();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Computed correlations for all " << num_users << " users" << std::endl;
        std::cout << "  Time: " << duration.count() << " ms" << std::endl;

        if (num_threads == 1) {
            double avg_corr = 0.0;
            int count = 0;
            for (int i = 0; i < std::min(5, num_users); i++) {
                for (int j = i+1; j < std::min(5, num_users); j++) {
                    avg_corr += correlations[i][j];
                    count++;
                }
            }
            if (count > 0) {
                std::cout << "  Sample correlations (User 0-4): " << std::fixed << std::setprecision(4) 
                          << (avg_corr/count) << " (avg)" << std::endl;
            }
        }
        std::cout << std::endl;

        // Test 2: Predict all ratings (PARALLEL)
        std::cout << "Test 2: Predict All Ratings (Parallel)" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        auto all_predictions = recommender.predictAllRatingsParallel(k_neighbors, correlations);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Predicted ratings for " << num_users << " users" << std::endl;
        std::cout << "  Time: " << duration.count() << " ms" << std::endl;

        if (num_threads == 1) {
            std::cout << "  Sample predictions (User 0, first 5 items): ";
            for (int i = 0; i < 5; i++) {
                std::cout << std::fixed << std::setprecision(2) << all_predictions[0][i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Test 3: Accuracy validation (PARALLEL)
        std::cout << "Test 3: Accuracy Validation (Parallel RMSE)" << std::endl;
        std::cout << "  Performing validation with 20% test set..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        double rmse = recommender.validateAccuracyParallel(0.2, k_neighbors);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  RMSE (Root Mean Square Error): " << std::fixed << std::setprecision(4) << rmse << std::endl;
        std::cout << "  Time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;

        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::endl;
    }

    // Performance analysis
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Performance Analysis" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::endl;

    std::cout << "Parallelization Strategy:" << std::endl;
    std::cout << "  1. Random data generation - Parallel for loop with thread-local RNG" << std::endl;
    std::cout << "  2. Correlation computation - Parallel for collapse(2) for user pairs" << std::endl;
    std::cout << "  3. Rating predictions - Parallel for over users" << std::endl;
    std::cout << "  4. RMSE calculation - Parallel for with reduction" << std::endl;
    std::cout << std::endl;

    std::cout << "Key Optimizations:" << std::endl;
    std::cout << "  - Correlation matrix computed once and reused" << std::endl;
    std::cout << "  - Dynamic scheduling for load balancing (variable computation per user)" << std::endl;
    std::cout << "  - Collapse(2) directive for better thread utilization" << std::endl;
    std::cout << "  - Reduction clause for thread-safe RMSE accumulation" << std::endl;
    std::cout << std::endl;

    std::cout << "Expected Speedup:" << std::endl;
    std::cout << "  - Ideal speedup = Number of threads" << std::endl;
    std::cout << "  - Actual speedup depends on:" << std::endl;
    std::cout << "    * Thread contention" << std::endl;
    std::cout << "    * Cache efficiency" << std::endl;
    std::cout << "    * Load imbalance" << std::endl;
    std::cout << "    * OpenMP overhead" << std::endl;
    std::cout << std::endl;

    std::cout << "Phase 2a Complete!" << std::endl;

    return 0;
}
