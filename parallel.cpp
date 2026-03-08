#include "parallel.h"
#include <random>
#include <ctime>

RecommendationSystemOpenMP::RecommendationSystemOpenMP(int users, int items, int threads)
    : num_users(users), num_items(items), rmse_error(0.0) {
    
    if (threads == -1) {
        num_threads = omp_get_num_procs();
    } else {
        num_threads = threads;
    }
    
    omp_set_num_threads(num_threads);
    ratings.resize(num_users, std::vector<double>(num_items, 0.0));
}

bool RecommendationSystemOpenMP::loadRatingsFromCSV(const std::string& filename) {
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

void RecommendationSystemOpenMP::generateRandomRatings(double sparsity) {
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < num_users; i++) {
        std::mt19937 gen(static_cast<unsigned>(time(0) + omp_get_thread_num() * 1000 + i));
        std::uniform_real_distribution<> dis(1.0, 5.0);
        std::uniform_real_distribution<> prob(0.0, 1.0);

        for (int j = 0; j < num_items; j++) {
            if (prob(gen) > sparsity) {
                ratings[i][j] = dis(gen);
            } else {
                ratings[i][j] = 0.0;
            }
        }
    }
}

double RecommendationSystemOpenMP::calculatePearsonCorrelation(int user1, int user2) {
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

std::vector<std::vector<double>> RecommendationSystemOpenMP::calculateAllCorrelations() {
    std::vector<std::vector<double>> correlations(num_users, std::vector<double>(num_users, 0.0));

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < num_users; i++) {
        for (int j = i; j < num_users; j++) {
            double corr = calculatePearsonCorrelation(i, j);
            correlations[i][j] = corr;
            correlations[j][i] = corr;  // Symmetric matrix
        }
    }

    return correlations;
}

std::vector<std::pair<int, double>> RecommendationSystemOpenMP::findNearestNeighbors(
    int user, int k, const std::vector<std::vector<double>>& correlations) {
    
    std::vector<std::pair<int, double>> neighbors;

    for (int i = 0; i < num_users; i++) {
        if (i != user) {
            neighbors.push_back({i, correlations[user][i]});
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

double RecommendationSystemOpenMP::predictRating(int user, int item, int k_neighbors,
                                                 const std::vector<std::vector<double>>& correlations) {
    if (ratings[user][item] > 0) {
        return ratings[user][item];
    }

    auto neighbors = findNearestNeighbors(user, k_neighbors, correlations);

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

std::vector<std::vector<double>> RecommendationSystemOpenMP::predictAllRatingsParallel(
    int k_neighbors, const std::vector<std::vector<double>>& correlations) {
    
    std::vector<std::vector<double>> predictions(num_users);

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int user = 0; user < num_users; user++) {
        predictions[user].resize(num_items);
        for (int item = 0; item < num_items; item++) {
            predictions[user][item] = predictRating(user, item, k_neighbors, correlations);
        }
    }

    return predictions;
}

double RecommendationSystemOpenMP::validateAccuracyParallel(double test_sparsity, int k_neighbors) {
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

    // Compute correlations once
    std::vector<std::vector<double>> correlations = calculateAllCorrelations();

    // Make predictions in parallel
    std::vector<double> predictions, actuals;
    predictions.resize(hidden_items.size());
    actuals.resize(hidden_items.size());

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t idx = 0; idx < hidden_items.size(); idx++) {
        int user = hidden_items[idx].first;
        int item = hidden_items[idx].second;
        predictions[idx] = predictRating(user, item, k_neighbors, correlations);
        actuals[idx] = original_ratings[user][item];
    }

    // Calculate RMSE
    rmse_error = calculateRMSE(predictions, actuals);

    // Restore original ratings
    ratings = original_ratings;

    return rmse_error;
}

double RecommendationSystemOpenMP::calculateRMSE(const std::vector<double>& predicted,
                                                  const std::vector<double>& actual) {
    if (predicted.size() != actual.size() || predicted.empty()) {
        return 0.0;
    }

    double sum_squared_errors = 0.0;

    #pragma omp parallel for reduction(+:sum_squared_errors) num_threads(num_threads)
    for (size_t i = 0; i < predicted.size(); i++) {
        double error = predicted[i] - actual[i];
        sum_squared_errors += error * error;
    }

    return std::sqrt(sum_squared_errors / predicted.size());
}

void RecommendationSystemOpenMP::printRatings() {
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
