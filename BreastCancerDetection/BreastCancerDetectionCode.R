options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
library(knitr)
library(rpart)
library(rpart.plot)

data(brca)

# Data exploration
dim(brca$x)[1]        # Number of samples
dim(brca$x)[2]        # Number of predictors
mean(brca$y == "M")   # Proportion malignant
which.max(colMeans(brca$x))     # Feature with highest mean
which.min(colSds(brca$x))       # Feature with lowest standard deviation

# Data normalization
x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

sd(x_scaled[, 1])     # Standard deviation of first column
median(x_scaled[, 1]) # Median of first column

# Distance analysis and clustering
d_samples <- dist(x_scaled)
dist_BtoB <- as.matrix(d_samples)[1, brca$y == "B"]
mean(dist_BtoB[2:length(dist_BtoB)])
dist_BtoM <- as.matrix(d_samples)[1, brca$y == "M"]
mean(dist_BtoM[2:length(dist_BtoM)])

d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)

# Principal Component Analysis (PCA)
prc <- prcomp(x_scaled)
summary(prc)

prc_explained <- cumsum(prc$sdev^2 / sum(prc$sdev^2))
plot(prc_explained, type = "l", col = "blue", lwd = 2,
     main = "Cumulative Explained Variance by Principal Components",
     xlab = "Principal Component", ylab = "Cumulative Variance Explained")

data.frame(prc$x[, 1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point() +
  labs(title = "PCA: First Two Principal Components")

data.frame(prc$x[, 1:10], type = brca$y) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, color = type)) +
  geom_boxplot() +
  labs(title = "Distribution of First 10 Principal Components by Tumor Type")

# Split into training and test sets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index, ]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index, ]
train_y <- brca$y[-test_index]

mean(train_y == "B")  # Proportion benign in training set
mean(test_y == "B")   # Proportion benign in test set

# K-Means Clustering
predict_kmeans <- function(x, k) {
  centers <- k$centers
  distances <- sapply(1:nrow(x), function(i) {
    apply(centers, 1, function(y) sqrt(sum((x[i, ] - y)^2)))
  })
  max.col(-t(distances))
}

train_x_scaled <- scale(train_x)
test_x_scaled <- scale(test_x)
results <- data.frame(Centroids = integer(), Accuracy = numeric())

for (centroids in 1:5) {
  set.seed(3, sample.kind = "Rounding")
  k <- kmeans(train_x_scaled, centers = centroids, iter.max = 100)
  kmeans_preds <- predict_kmeans(test_x_scaled, k)
  kmeans_preds <- ifelse(kmeans_preds == 2, "B", "M")
  accuracy <- mean(kmeans_preds == test_y)
  results <- rbind(results, data.frame(Centroids = centroids, Accuracy = accuracy))
}

plot(results$Centroids, results$Accuracy, type = "o", pch = 16, lwd = 2, col = "darkblue",
     xlab = "Number of Centroids", ylab = "Accuracy",
     main = "K-Means Clustering Accuracy vs Number of Centroids")
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")

set.seed(3, sample.kind = "Rounding")
k <- kmeans(train_x, centers = 2)
best_kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")
accuracy_table <- data.frame(Model = "K-Means", Accuracy = mean(best_kmeans_preds == test_y))

# Logistic Regression (with Lasso & Ridge)
set.seed(1, sample.kind = "Rounding")
train_glm <- train(train_x, train_y, method = "glm")
glm_preds <- predict(train_glm, test_x)
mean(glm_preds == test_y)

train_glmnet <- train(train_x, train_y,
                      method = "glmnet",
                      tuneGrid = expand.grid(alpha = c(0, 1),
                                             lambda = seq(0.001, 0.1, length = 10)))
results <- train_glmnet$results

ggplot(results, aes(x = lambda, y = Accuracy, color = factor(alpha))) +
  geom_point(size = 3) +
  geom_line(size = 1.5) +
  labs(title = "Accuracy vs Lambda (Lasso & Ridge)",
       x = "Lambda", y = "Accuracy", color = "Regularization") +
  scale_color_manual(values = c("darkblue", "darkorange"),
                     labels = c("Ridge (α=0)", "Lasso (α=1)")) +
  theme_minimal(base_size = 15)

best_ridge <- results[results$alpha == 0, ][which.max(results$Accuracy[results$alpha == 0]), ]
best_lasso <- results[results$alpha == 1, ][which.max(results$Accuracy[results$alpha == 1]), ]

accuracy_table <- rbind(accuracy_table,
                        data.frame(Model = "Logistic Regression (Ridge)", Accuracy = best_ridge$Accuracy),
                        data.frame(Model = "Logistic Regression (Lasso)", Accuracy = best_lasso$Accuracy))

# K-Nearest Neighbors (KNN)
set.seed(7, sample.kind = "Rounding")
tuning <- data.frame(k = seq(3, 21, 2))
train_knn <- train(train_x, train_y,
                   method = "knn",
                   tuneGrid = tuning,
                   trControl = trainControl(method = "cv", number = 10))
knn_preds <- predict(train_knn, test_x)

accuracy_results <- train_knn$results
best_k <- accuracy_results$k[which.max(accuracy_results$Accuracy)]
best_accuracy <- max(accuracy_results$Accuracy)

plot(accuracy_results$k, accuracy_results$Accuracy, type = "o", pch = 16, lwd = 2, col = "blue",
     xlab = "Number of Neighbors (k)", ylab = "Accuracy",
     main = "KNN Accuracy vs Number of Neighbors")
grid(nx = NA, ny = NULL, col = "gray", lty = "dotted")
points(best_k, best_accuracy, col = "red", pch = 19, cex = 2)
abline(v = best_k, col = "red", lty = "dashed")
text(best_k, best_accuracy, labels = paste("Best k =", best_k, "\nAccuracy =", round(best_accuracy * 100, 2), "%"),
     pos = 3, col = "red", cex = 1)

accuracy_table <- rbind(accuracy_table,
                        data.frame(Model = "K-Nearest Neighbors", Accuracy = best_accuracy))

# Random Forest
set.seed(9, sample.kind = "Rounding")
tuning <- data.frame(mtry = 1:9)
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = tuning,
                  importance = TRUE)
rf_preds <- predict(train_rf, test_x)

rf_results <- train_rf$results
best_mtry <- rf_results$mtry[which.max(rf_results$Accuracy)]
best_accuracy <- max(rf_results$Accuracy)

ggplot(rf_results, aes(x = mtry, y = Accuracy)) +
  geom_point(size = 3, color = "darkorange") +
  geom_line(color = "darkgreen", size = 1.5) +
  geom_point(aes(x = best_mtry, y = best_accuracy), color = "red", size = 4) +
  geom_vline(xintercept = best_mtry, linetype = "dashed", color = "red") +
  annotate("text", x = best_mtry, y = best_accuracy + 0.002,
           label = paste("Best mtry =", best_mtry, "\nAccuracy =", round(best_accuracy * 100, 2), "%"),
           color = "red", size = 5, vjust = -0.5) +
  labs(title = "Random Forest Accuracy vs mtry",
       x = "mtry (Number of Predictors)", y = "Accuracy") +
  theme_minimal(base_size = 15)

rf_accuracy <- max(rf_results$Accuracy)
accuracy_table <- rbind(accuracy_table,
                        data.frame(Model = "Random Forest", Accuracy = rf_accuracy))

fit <- rpart(train_y ~ ., data = data.frame(train_x, train_y))
custom_palette <- c("#66C2A5", "#FC8D62")
rpart.plot(fit, type = 3, extra = 101, under = TRUE, faclen = 0,
           tweak = 1.2, box.palette = custom_palette,
           shadow.col = "gray", fallen.leaves = TRUE)

# Final Model Accuracy Summary
kable(accuracy_table, caption = "Accuracy of Machine Learning Algorithms")