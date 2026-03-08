#include "core.h"
#include <chrono>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Amazon Dataset - Serial Recommendation System ===" << std::endl;
    std::cout << std::endl;

    // Parameters - adjust based on your dataset
    int num_users = 100;
    int num_items = 500;
    int k_neighbors = 10;

    std::cout << "Dataset Configuration:" << std::endl;
    std::cout << "  File: amazon_ratings.csv" << std::endl;
    std::cout << "  Users: " << num_users << std::endl;
    std::cout << "  Items: " << num_items << std::endl;
    std::cout << "  K-neighbors: " << k_neighbors << std::endl;
    std::cout << std::endl;

    // Initialize recommendation system
    RecommendationSystem recommender(num_users, num_items);

    // Try to load from CSV file, fallback to random if file not found
    std::cout << "Loading ratings from CSV..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!recommender.loadRatingsFromCSV("amazon_ratings.csv")) {
        std::cout << "Note: amazon_ratings.csv not found. Using generated data instead." << std::endl;
        std::cout << "To use real data:" << std::endl;
        std::cout << "  1. Run: python download_amazon_dataset.py" << std::endl;
        std::cout << "  2. Place amazon_ratings.csv in the same directory" << std::endl;
        std::cout << std::endl;
        recommender.generateRandomRatings(0.8);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    // Display sample of ratings
    std::cout << "Sample of Ratings Matrix:" << std::endl;
    recommender.printRatings();
    std::cout << std::endl;

    // Test 1: Calculate correlations
    std::cout << "Test 1: Pearson Correlation Calculation" << std::endl;
    std::cout << "  Correlations between User 0 and other users:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 5; i++) {
        double corr = recommender.calculatePearsonCorrelation(0, i);
        std::cout << "    User 0 <-> User " << i << ": " << std::fixed << std::setprecision(4) << corr << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    // Test 2: Find nearest neighbors
    std::cout << "Test 2: Finding Nearest Neighbors" << std::endl;
    std::cout << "  Top " << k_neighbors << " neighbors of User 0:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto neighbors = recommender.findNearestNeighbors(0, k_neighbors);
    for (size_t i = 0; i < neighbors.size(); i++) {
        std::cout << "    " << (i+1) << ". User " << neighbors[i].first 
                  << " (correlation: " << std::fixed << std::setprecision(4) 
                  << neighbors[i].second << ")" << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    // Test 3: Rating prediction for a single item
    std::cout << "Test 3: Single Item Rating Prediction" << std::endl;
    int test_user = 5;
    int test_item = 42;
    std::cout << "  Predicting rating for User " << test_user << ", Item " << test_item << std::endl;
    start = std::chrono::high_resolution_clock::now();
    double predicted = recommender.predictRating(test_user, test_item, k_neighbors);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    Predicted rating: " << std::fixed << std::setprecision(4) << predicted << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    // Test 4: Predictions for all items of a user
    std::cout << "Test 4: All Items Rating Prediction" << std::endl;
    std::cout << "  Predicting ratings for all items of User " << test_user << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto predictions = recommender.predictAllRatings(test_user, k_neighbors);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    Sample predictions (first 10 items):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "      Item " << i << ": " << std::fixed << std::setprecision(4) 
                  << predictions[i] << std::endl;
    }
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    // Test 5: Accuracy validation (RMSE)
    std::cout << "Test 5: Accuracy Validation (RMSE)" << std::endl;
    std::cout << "  Performing validation with 20% test set..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    double rmse = recommender.validateAccuracy(0.2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    RMSE (Root Mean Square Error): " << std::fixed << std::setprecision(4) << rmse << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    // Summary
    std::cout << "=== Performance Summary ===" << std::endl;
    std::cout << "System: Serial Baseline (Amazon Dataset)" << std::endl;
    std::cout << "Algorithm: Pearson Correlation with User-Based Collaborative Filtering" << std::endl;
    std::cout << "Accuracy (RMSE): " << std::fixed << std::setprecision(4) << rmse << std::endl;
    std::cout << std::endl;
    std::cout << "Phase Complete!" << std::endl;

    return 0;
}
