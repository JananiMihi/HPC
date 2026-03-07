#ifndef RECOMMENDATION_H
#define RECOMMENDATION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>

typedef std::vector<std::vector<double>> Matrix;

class RecommendationSystem {
private:
    Matrix ratings;          // User-item ratings matrix (users x items)
    int num_users;
    int num_items;
    double rmse_error;

public:
    RecommendationSystem(int users, int items);
    
    // Load ratings from CSV file
    bool loadRatingsFromCSV(const std::string& filename);
    
    // Generate random ratings for testing
    void generateRandomRatings(double sparsity = 0.8);
    
    // Calculate Pearson correlation between two users
    double calculatePearsonCorrelation(int user1, int user2);
    
    // Find k nearest neighbors for a user
    std::vector<std::pair<int, double>> findNearestNeighbors(int user, int k);
    
    // Predict rating for an item using collaborative filtering
    double predictRating(int user, int item, int k_neighbors = 5);
    
    // Make predictions for all unrated items of a user
    std::vector<double> predictAllRatings(int user, int k_neighbors = 5);
    
    // Validate accuracy compared to held-out test set
    double validateAccuracy(double test_sparsity = 0.2);
    
    // Calculate RMSE (Root Mean Square Error)
    double calculateRMSE(const std::vector<double>& predicted, 
                         const std::vector<double>& actual);
    
    // Print ratings matrix
    void printRatings();
    
    // Getters
    int getNumUsers() const { return num_users; }
    int getNumItems() const { return num_items; }
    double getRating(int user, int item) const { return ratings[user][item]; }
    double getRMSE() const { return rmse_error; }
    
    // Setters
    void setRating(int user, int item, double rating) {
        ratings[user][item] = rating;
    }
};

#endif // RECOMMENDATION_H
