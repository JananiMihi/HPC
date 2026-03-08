#include "core.h"
#include <random>
#include <ctime>

RecommendationSystem::RecommendationSystem(int users, int items)
    : num_users(users), num_items(items), rmse_error(0.0) {
    ratings.resize(num_users, std::vector<double>(num_items, 0.0));
}

bool RecommendationSystem::loadRatingsFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int user = 0;
    while (std::getline(file, line) && user < num_users) {
        std::stringstream ss(line);
        std::string value;
        int item = 0;
        while (std::getline(ss, value, ',') && item < num_items) {
            try {
                ratings[user][item] = std::stod(value);
                item++;
            } catch (...) {
                std::cerr << "Error parsing value: " << value << std::endl;
                return false;
            }
        }
        user++;
    }
    file.close();
    return true;
}

void RecommendationSystem::generateRandomRatings(double sparsity) {
    std::mt19937 gen(static_cast<unsigned>(time(0)));
    std::uniform_real_distribution<> dis(1.0, 5.0);  // Ratings 1-5
    std::uniform_real_distribution<> prob(0.0, 1.0);

    for (int i = 0; i < num_users; i++) {
        for (int j = 0; j < num_items; j++) {
            if (prob(gen) > sparsity) {
                ratings[i][j] = dis(gen);
            } else {
                ratings[i][j] = 0.0;  // 0 means not rated
            }
        }
    }
}

double RecommendationSystem::calculatePearsonCorrelation(int user1, int user2) {
    if (user1 == user2) return 1.0;

    // Find common rated items
    std::vector<double> ratings1, ratings2;
    for (int item = 0; item < num_items; item++) {
        if (ratings[user1][item] > 0 && ratings[user2][item] > 0) {
            ratings1.push_back(ratings[user1][item]);
            ratings2.push_back(ratings[user2][item]);
        }
    }

    // Need at least 2 common ratings
    if (ratings1.size() < 2) return 0.0;

    // Calculate means
    double mean1 = std::accumulate(ratings1.begin(), ratings1.end(), 0.0) / ratings1.size();
    double mean2 = std::accumulate(ratings2.begin(), ratings2.end(), 0.0) / ratings2.size();

    // Calculate Pearson correlation
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

std::vector<std::pair<int, double>> RecommendationSystem::findNearestNeighbors(int user, int k) {
    std::vector<std::pair<int, double>> neighbors;

    // Calculate correlation with all other users
    for (int i = 0; i < num_users; i++) {
        if (i != user) {
            double correlation = calculatePearsonCorrelation(user, i);
            neighbors.push_back({i, correlation});
        }
    }

    // Sort by correlation in descending order
    std::sort(neighbors.begin(), neighbors.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });

    // Keep only top k neighbors
    if (neighbors.size() > static_cast<size_t>(k)) {
        neighbors.resize(k);
    }

    return neighbors;
}

double RecommendationSystem::predictRating(int user, int item, int k_neighbors) {
    // If user has already rated this item, return the rating
    if (ratings[user][item] > 0) {
        return ratings[user][item];
    }

    // Find k nearest neighbors
    auto neighbors = findNearestNeighbors(user, k_neighbors);

    // Calculate weighted average of neighbor ratings
    double weighted_sum = 0.0;
    double weight_sum = 0.0;

    for (const auto& neighbor : neighbors) {
        int neighbor_id = neighbor.first;
        double correlation = neighbor.second;

        // Only consider neighbors who have rated this item
        if (ratings[neighbor_id][item] > 0) {
            weighted_sum += correlation * ratings[neighbor_id][item];
            weight_sum += std::abs(correlation);
        }
    }

    if (weight_sum < 1e-10) {
        return 3.0;  // Default to middle rating if no neighbors rated it
    }

    double predicted = weighted_sum / weight_sum;
    // Clamp to valid rating range [1, 5]
    return std::max(1.0, std::min(5.0, predicted));
}

std::vector<double> RecommendationSystem::predictAllRatings(int user, int k_neighbors) {
    std::vector<double> predictions;
    for (int item = 0; item < num_items; item++) {
        predictions.push_back(predictRating(user, item, k_neighbors));
    }
    return predictions;
}

double RecommendationSystem::validateAccuracy(double test_sparsity) {
    // Create a test set by hiding random ratings
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

    // Make predictions for hidden items
    std::vector<double> predictions, actuals;
    for (const auto& item : hidden_items) {
        int user = item.first;
        int item_id = item.second;
        double predicted = predictRating(user, item_id);
        double actual = original_ratings[user][item_id];

        predictions.push_back(predicted);
        actuals.push_back(actual);
    }

    // Calculate RMSE
    rmse_error = calculateRMSE(predictions, actuals);

    // Restore original ratings
    ratings = original_ratings;

    return rmse_error;
}

double RecommendationSystem::calculateRMSE(const std::vector<double>& predicted,
                                            const std::vector<double>& actual) {
    if (predicted.size() != actual.size() || predicted.empty()) {
        return 0.0;
    }

    double sum_squared_errors = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        double error = predicted[i] - actual[i];
        sum_squared_errors += error * error;
    }

    return std::sqrt(sum_squared_errors / predicted.size());
}

void RecommendationSystem::printRatings() {
    std::cout << "Ratings Matrix (" << num_users << " users x " << num_items << " items):" << std::endl;
    for (int i = 0; i < std::min(num_users, 5); i++) {
        for (int j = 0; j < std::min(num_items, 10); j++) {
            printf("%6.2f ", ratings[i][j]);
        }
        if (num_items > 10) std::cout << "...";
        std::cout << std::endl;
    }
    if (num_users > 5) std::cout << "..." << std::endl;
}
