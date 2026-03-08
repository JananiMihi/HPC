#include "parallel.h"
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
    printHeader("OpenMP Parallelization - Amazon Dataset Recommendation System");

    // Parameters
    int num_users = 100;
    int num_items = 500;
    int k_neighbors = 10;

    std::cout << "Dataset Configuration:" << std::endl;
    std::cout << "  File: amazon_ratings.csv" << std::endl;
    std::cout << "  Users: " << num_users << std::endl;
    std::cout << "  Items: " << num_items << std::endl;
    std::cout << "  K-neighbors: " << k_neighbors << std::endl;
    std::cout << std::endl;

    int num_ranks;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_ranks = omp_get_num_threads();
        }
    }

    std::cout << "OpenMP Configuration:" << std::endl;
    std::cout << "  Max threads available: " << omp_get_num_procs() << std::endl;
    std::cout << std::endl;

    // Test with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    for (int num_threads : thread_counts) {
        if (num_threads > omp_get_num_procs()) {
            std::cout << "Skipping " << num_threads << " threads (not available)" << std::endl;
            continue;
        }

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Running with " << num_threads << " thread(s)" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        RecommendationSystemOpenMP recommender(num_users, num_items, num_threads);

        // Load or generate ratings
        std::cout << "Loading ratings from CSV..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!recommender.loadRatingsFromCSV("amazon_ratings.csv")) {
            std::cout << "Using generated data (amazon_ratings.csv not found)" << std::endl;
            recommender.generateRandomRatings(0.8);
        }
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

        // Test 3: Validate accuracy (PARALLEL)
        std::cout << "Test 3: Validate Accuracy (Parallel)" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        double rmse = recommender.validateAccuracyParallel(0.2, k_neighbors);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  RMSE: " << std::fixed << std::setprecision(4) << rmse << std::endl;
        std::cout << "  Time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;
    }

    // Print summary
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  This OpenMP implementation parallelizes correlation calculation" << std::endl;
    std::cout << "  and prediction across multiple threads for improved performance." << std::endl;
    std::cout << "  Dataset: Amazon Reviews (Real-world collaborative filtering data)" << std::endl;
    std::cout << std::endl;

    return 0;
}
