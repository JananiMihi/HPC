/*
 * Single-file implementation for Phase 1: Serial Baseline
 * Pearson Correlation Based Recommendation System
 * Compile with: g++ -std=c++17 -O2 -o phase1_serial phase1_serial.cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>
#include <iomanip>
#include <chrono>

typedef std::vector<std::vector<double>> Matrix;

class RecommendationSystem {
private:
    Matrix ratings;
    int num_users;
    int num_items;
    double rmse_error;

public:
    RecommendationSystem(int users, int items)
        : num_users(users), num_items(items), rmse_error(0.0) {
        ratings.resize(num_users, std::vector<double>(num_items, 0.0));
    }

    void generateRandomRatings(double sparsity = 0.8) {
        std::mt19937 gen(static_cast<unsigned>(time(0)));
        std::uniform_real_distribution<> dis(1.0, 5.0);
        std::uniform_real_distribution<> prob(0.0, 1.0);

        for (int i = 0; i < num_users; i++) {
            for (int j = 0; j < num_items; j++) {
                if (prob(gen) > sparsity) {
                    ratings[i][j] = dis(gen);
                } else {
                    ratings[i][j] = 0.0;
                }
            }
        }
    }

    double calculatePearsonCorrelation(int user1, int user2) {
        if (user1 == user2) return 1.0;

        std::vector<double> ratings1, ratings2;
        for (int item = 0; item < num_items; item++) {
            if (ratings[user1][item] > 0 && ratings[user2][item] > 0) {
                ratings1.push_back(ratings[user1][item]);
                ratings2.push_back(ratings[user2][item]);
            }
        }

        if (ratings1.size() < 2) return 0.0;

        double mean1 = std::accumulate(ratings1.begin(), ratings1.end(), 0.0) / ratings1.size();
        double mean2 = std::accumulate(ratings2.begin(), ratings2.end(), 0.0) / ratings2.size();

        double numerator = 0.0;
        double denominator1 = 0.0;
        double denominator2 = 0.0;

        for (size_t i = 0; i < ratings1.size(); i++) {
            double diff1 = ratings1[i] - mean1;
            double diff2 = ratings2[i] - mean2;
            numerator += diff1 * diff2;
            denominator1 += diff1 * diff1;
            denominator2 += diff2 * diff2;
        }

        double denominator = std::sqrt(denominator1 * denominator2);
        if (denominator < 1e-10) return 0.0;

        return numerator / denominator;
    }

    std::vector<std::pair<int, double>> findNearestNeighbors(int user, int k) {
        std::vector<std::pair<int, double>> neighbors;

        for (int i = 0; i < num_users; i++) {
            if (i != user) {
                double correlation = calculatePearsonCorrelation(user, i);
                neighbors.push_back({i, correlation});
            }
        }

        std::sort(neighbors.begin(), neighbors.end(),
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });

        if (neighbors.size() > static_cast<size_t>(k)) {
            neighbors.resize(k);
        }

        return neighbors;
    }

    double predictRating(int user, int item, int k_neighbors = 5) {
        if (ratings[user][item] > 0) {
            return ratings[user][item];
        }

        auto neighbors = findNearestNeighbors(user, k_neighbors);

        double weighted_sum = 0.0;
        double weight_sum = 0.0;

        for (const auto& neighbor : neighbors) {
            int neighbor_id = neighbor.first;
            double correlation = neighbor.second;

            if (ratings[neighbor_id][item] > 0) {
                weighted_sum += correlation * ratings[neighbor_id][item];
                weight_sum += std::abs(correlation);
            }
        }

        if (weight_sum < 1e-10) {
            return 3.0;
        }

        double predicted = weighted_sum / weight_sum;
        return std::max(1.0, std::min(5.0, predicted));
    }

    std::vector<double> predictAllRatings(int user, int k_neighbors = 5) {
        std::vector<double> predictions;
        for (int item = 0; item < num_items; item++) {
            predictions.push_back(predictRating(user, item, k_neighbors));
        }
        return predictions;
    }

    double validateAccuracy(double test_sparsity = 0.2) {
        Matrix original_ratings = ratings;
        std::vector<std::pair<int, int>> hidden_items;

        std::mt19937 gen(static_cast<unsigned>(time(0)) + 1);
        std::uniform_real_distribution<> prob(0.0, 1.0);

        for (int i = 0; i < num_users; i++) {
            for (int j = 0; j < num_items; j++) {
                if (ratings[i][j] > 0 && prob(gen) < test_sparsity) {
                    hidden_items.push_back({i, j});
                    ratings[i][j] = 0.0;
                }
            }
        }

        std::vector<double> predictions, actuals;
        for (const auto& item : hidden_items) {
            int user = item.first;
            int item_id = item.second;
            double predicted = predictRating(user, item_id);
            double actual = original_ratings[user][item_id];

            predictions.push_back(predicted);
            actuals.push_back(actual);
        }

        if (!predictions.empty()) {
            double sum_squared_errors = 0.0;
            for (size_t i = 0; i < predictions.size(); i++) {
                double error = predictions[i] - actuals[i];
                sum_squared_errors += error * error;
            }
            rmse_error = std::sqrt(sum_squared_errors / predictions.size());
        }

        ratings = original_ratings;

        return rmse_error;
    }

    void printRatings() {
        std::cout << "Ratings Matrix (" << num_users << " users x " << num_items << " items):" << std::endl;
        for (int i = 0; i < std::min(num_users, 5); i++) {
            for (int j = 0; j < std::min(num_items, 10); j++) {
                std::cout << std::fixed << std::setw(6) << std::setprecision(2) << ratings[i][j] << " ";
            }
            if (num_items > 10) std::cout << "...";
            std::cout << std::endl;
        }
        if (num_users > 5) std::cout << "..." << std::endl;
    }

    int getNumUsers() const { return num_users; }
    int getNumItems() const { return num_items; }
    double getRMSE() const { return rmse_error; }
};

int main() {
    std::cout << "===  Serial Baseline - Pearson Correlation Recommendation System ===" << std::endl;
    std::cout << std::endl;

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

    RecommendationSystem recommender(num_users, num_items);

    std::cout << "Generating random ratings..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    recommender.generateRandomRatings(sparsity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "Sample of Ratings Matrix:" << std::endl;
    recommender.printRatings();
    std::cout << std::endl;

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

    std::cout << "Test 5: Accuracy Validation (RMSE)" << std::endl;
    std::cout << "  Performing validation with 20% test set..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    double rmse = recommender.validateAccuracy(0.2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    RMSE (Root Mean Square Error): " << std::fixed << std::setprecision(4) << rmse << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Performance Summary ===" << std::endl;
    std::cout << "System: Serial Baseline" << std::endl;
    std::cout << "Algorithm: Pearson Correlation with User-Based Collaborative Filtering" << std::endl;
    std::cout << "Accuracy (RMSE): " << std::fixed << std::setprecision(4) << rmse << std::endl;
    std::cout << std::endl;

    std::cout << "Complete!" << std::endl;

    return 0;
}
