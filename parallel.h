#ifndef RECOMMENDATION_OPENMP_H
#define RECOMMENDATION_OPENMP_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <omp.h>

typedef std::vector<std::vector<double>> Matrix;

class RecommendationSystemOpenMP {
private:
    Matrix ratings;          // User-item ratings matrix (users x items)
    int num_users;
    int num_items;
    int num_threads;
    double rmse_error;

public:
    RecommendationSystemOpenMP(int users, int items, int threads = -1);
    
    // Load ratings from CSV file
    bool loadRatingsFromCSV(const std::string& filename);
    
    // Generate random ratings for testing
    void generateRandomRatings(double sparsity = 0.8);
    
    // Calculate Pearson correlation between two users (serial)
    double calculatePearsonCorrelation(int user1, int user2);
    
    // Calculate correlations for all users (parallel)
    std::vector<std::vector<double>> calculateAllCorrelations();
    
    // Find k nearest neighbors for a user (using pre-computed correlations)
    std::vector<std::pair<int, double>> findNearestNeighbors(int user, int k,
                                                             const std::vector<std::vector<double>>& correlations);
    
    // Predict rating for an item using collaborative filtering
    double predictRating(int user, int item, int k_neighbors, 
                         const std::vector<std::vector<double>>& correlations);
    
    // Make predictions for all unrated items of users (parallel)
    std::vector<std::vector<double>> predictAllRatingsParallel(int k_neighbors,
                                                               const std::vector<std::vector<double>>& correlations);
    
    // Validate accuracy compared to held-out test set (parallel)
    double validateAccuracyParallel(double test_sparsity = 0.2, int k_neighbors = 5);
    
    // Calculate RMSE (Root Mean Square Error)
    double calculateRMSE(const std::vector<double>& predicted, 
                         const std::vector<double>& actual);
    
    // Print ratings matrix
    void printRatings();
    
    // Getters
    int getNumUsers() const { return num_users; }
    int getNumItems() const { return num_items; }
    int getNumThreads() const { return num_threads; }
    double getRating(int user, int item) const { return ratings[user][item]; }
    double getRMSE() const { return rmse_error; }
    
    // Setters
    void setRating(int user, int item, double rating) {
        ratings[user][item] = rating;
    }
};

#endif // RECOMMENDATION_OPENMP_H
