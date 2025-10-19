# This is the main R Script that has the algorithm that I made in order to predict movie ratings
# First we will begin by downloading our data sets from edx

########################################
# Create edx and final_holdout_test sets
########################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")


# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#####################################
# We will start by exploring our Data
#####################################

# First we look at the first 5 columns in tidy format
tibble(head(edx, 5))

# See the dimensions of our data
dim(edx)

# Then we can see a summary of our data as well to get a more specific understanding
summary(edx)

# To get a better understanding we should see the number of different users, movies, and generes

edx %>%
  summarize(number_of_users = n_distinct(userId),
            number_of_movies = n_distinct(movieId),
            number_of_ratings = n_distinct(genres))

# We have to look at popularity
# Let's see the rating of the genres and which is the most popular.
genres <- c("Comedy", "Drama", "Romance", "Thriller")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# We can also see the 10 movies with the highest number of ratings
edx %>%
  group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(10)

# Now we will create our training and test sets with the test sets being 20% of the data
set.seed(1, sample.kind="Rounding")
test_set_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_set_index,]
test_set <- edx[test_set_index,]

# We will improve our data by making sure that there aren't any users in the test set that do not
# appear in the train set or vice-versa
test_set <- test_set %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "movieId")

# Now we will build our Residual Mean Squared Error function

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



############################## 
# Building the Model
############################## 


# Model 1
# Let's start by trying to predict the ratings with the simplest way possible through making the 
# mu variable equal to the average value of the ratings 
mu <- mean(train_set$rating)

# Let's see what RMSE we get
mu_rmse <- RMSE(test_set$rating, mu)
mu_rmse

# We obtain an rmse of 1.06
# We want to create a table that will store all our different methods and RMSE's
rmse_model_results <- tibble(method = "The Average of Ratings", rmse = mu_rmse)


# Model 2
#### The next term we can add is something that will take the genres into consideration.
genre_averages <- train_set %>% 
  group_by(genres) %>%
  summarize(g_i = mean(rating))

# We will use the variable g_i in order to predict the ratings.
genre_average_pred <- test_set %>% 
  mutate(Match = match(genres, genre_averages$genres), prediction = genre_averages$g_i[Match]) %>%
  pull(prediction)


# Now let's see the results of our new model

genre_average_rmse <- RMSE(test_set$rating, genre_average_pred)
genre_average_rmse

# We will save that into our table

rmse_model_results <- bind_rows(rmse_model_results, tibble(method = "The Genre Rating Effect",
                                                           rmse = genre_average_rmse))

# We do see that our rmse decreased to to  1.018
# Now let's add more terms in our model to obtain a lower RMSE

# Model 3
# First we will add the term b_i which will be the difference between a movie and the
# average rating for every movie. We have already calculated the average rating for every movie

movie_averages <- train_set %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Now let's see the results of our new model

movie_averages_pred <- mu + test_set %>%
  left_join(movie_averages, by = 'movieId') %>%
  pull(b_i)

movie_average_rmse <- RMSE(test_set$rating, movie_averages_pred)
movie_average_rmse

# Let's save that into our table
rmse_model_results <- bind_rows(rmse_model_results, tibble(method = "The Movie Rating Effect",
                                                           rmse = movie_average_rmse))

# The new RMSE we get is 0.94 which is an improvement but we still need to get better.
# Let's look more deeper into the data set 

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black", fill = "light blue")


# From this we can infer that there needs to be a user effect and we will create one called b_u

user_averages <- train_set %>% 
  left_join(movie_averages, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Model 4
# Hence the RMSE of our new model will be the following

user_average_pred <- test_set %>% 
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)


user_averages_rmse <- RMSE(test_set$rating, user_average_pred)
user_averages_rmse

# Let's save that into our table
rmse_model_results <- bind_rows(rmse_model_results,
                                tibble(method = "The Movie and User Rating Effect",
                                       rmse = user_averages_rmse))

# The RMSE is 0.866
# We can see that the RMSE has dropped significantly after this feature.


# However we still need to improve. To do this we need to further analyse our predictions
# We will look at the 10 largest mistakes that we made

test_set %>%
  left_join(movie_averages, by = "movieId") %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%
  select(title, residual) %>%
  slice(1:10) %>%
  knitr::kable()

# These are all uncertain movies and for many of them our model obtained large predictions 
# We need to further look into the 10 best movies and the 10 worst movies 
# To do this we need to create a database with the movie id's and movie titles

movie_titles <- edx %>%
  select(movieId, title) %>%
  distinct()

# Here are the top 10 best movies 

movie_averages %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i) %>%
  slice(1:10) %>%
  knitr::kable()

# Here are the top 10 worst movies 

movie_averages %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(b_i) %>%
  select(title, b_i) %>%
  slice(1:10) %>%
  knitr::kable()


# We can see that they are all quite obscure movies, so the next thing we can do is see 
# the number of ratings each received

train_set %>%
  dplyr::count(movieId) %>%
  left_join(movie_averages) %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>%
  knitr::kable()

# We can clearly see that the best and worst movies in-fact have extremely low ratings.
# We can see that larger estimates of b_i result, when there are fewer ratings. They are noisy 
# estimates that we should constrain using regularization. 


# We will find the lambda that best regularizes the users and movies

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, test_set$rating))
})


# Model 5
# We can see that the best choice for Lambda is 4.75 through the following code.

lambda <- lambdas[which.min(rmses)]
lambda

# We can also see it through the a graph
qplot(lambdas, rmses)


# Now we will use this lambda to re-compute our estimates of b_i

mu <- mean(train_set$rating)
movie_reg_averages <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())


# Now we will check to see if our results improved.

movie_reg_average_pred <- test_set %>% 
  left_join(movie_reg_averages, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)


movie_reg_average_rmse <- RMSE(movie_reg_average_pred, test_set$rating)
movie_reg_average_rmse


rmse_model_results <- bind_rows(rmse_model_results,
                                tibble(method = "The Regularized Movie Rating Effect",
                                       rmse = movie_reg_average_rmse))
rmse_model_results

# Model 6
# We see that if we compare the movie rating effect without the regularization and after the 
# regularization it droops from 0.9437 to 0.9436
# So we also have to regularize the user rating effect as well

user_reg_averages <- train_set %>%
  left_join(movie_reg_averages, by = 'movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_i = n()) 

# We get the predictions
user_reg_averages_pred <- test_set %>%
  left_join(movie_reg_averages, by = 'movieId') %>%
  left_join(user_reg_averages, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# We find the rmse
user_reg_averages_rmse <- RMSE(user_reg_averages_pred, test_set$rating)
user_reg_averages_rmse

# We will now add the rmse to our table 

rmse_model_results <- bind_rows(rmse_model_results,
                                tibble(method = "The Regularized Movie and User Rating Effect",
                                       rmse = user_reg_averages_rmse))

# We can see a descending list of our RMSE's with the different models we used
rmse_model_results %>% arrange(desc(rmse))



# Model 7 & 8
# Now we will use Matrix Factorization to try and decrease our rmse

# We load the library required for the matrix factorization
library(recosystem)

# We set the seed to 1
set.seed(1, sample.kind = "Rounding")

# Now we will make the training and test sets into the format required by the 
# recosystem library.

train_set_reco <-  with(train_set, data_memory(user_index = userId, 
                                               item_index = movieId,
                                               rating     = rating,
                                               dates     = dates))

test_set_reco  <-  with(test_set,  data_memory(user_index = userId, 
                                               item_index = movieId, 
                                               rating     = rating,
                                               dates     = dates))


# Now we create the 2 models and store them in an object
recosystem_model_1 <-  recosystem::Reco()
recosystem_model_2 <-  recosystem::Reco()


# Then we will chose the tuning parameters for our model. The parameters I used were from different
# examples of people using reco from rdrr.io and are considered the default tuning parameters.

# This is the first set of parameters that we will try and save it to a variable called optimization_1
# Please keep in mind that it will take a long time and use up a lot of computing power. 
######### WARNING: The next line of code with optimization_1 took around 30 minutes to optimize!!

optimization_1 <- recosystem_model_1$tune(train_set_reco, 
                                          opts = list(dim = c(10, 20, 30),          # dim is the number of latent factors.
                                                      lrate = c(0.1, 0.2),          # lrate is the learning rate.
                                                      costp_l2 = c(0.01, 0.1),      # costp_l2 is the L2 regularization cost for the user factors.
                                                      costq_l2 = c(0.01, 0.1),      # costq_l2 is the L2 regularization cost for the item factors.
                                                      nthread  = 4,                 # nthread is an integer and is the number of threads for parallel computing.
                                                      niter = 20))                  # niter is an integer and is the number of iterations.


# This is the first set of parameters that we will try and save it to a variable called optimization_2
# Please keep in mind that it will take a long time and use up a lot of computing power. 
######### WARNING: The next line of code with optimization_2 took around 1:30 hours to optimize!!

optimization_2 <- recosystem_model_2$tune(train_set_reco, opts = list(dim = c(10L, 20L), # dim is the number of latent factors.
                                                                      costp_l1 = c(0, 0.1),       # costp_l1 is the l1 regularization cost for user factors. 
                                                                      costp_l2 = c(0.01, 0.1),    # costp_l2 is the l2 regularization cost for user factors.
                                                                      costq_l1 = c(0, 0.1),       # costq_l1 is the l1 regularization cost for item factors.
                                                                      costq_l2 = c(0.01, 0.1),    # costq_l2 is the l2 regularization cost for item factors.
                                                                      lrate    = c(0.01, 0.1))    # lrate is the learning rate.
)

# Then we will train the algorithm through selecting the train in our models defined above.
# optimization_1:

recosystem_model_1$train(train_set_reco,
                         opts = c(optimization_1$min))

# optimization_2:
recosystem_model_2$train(train_set_reco, opts = optimization_2$min) 



# Now we will find the predicted values using the test_data that we defined 
# out_memory() returns the result as an R object
# optimization_1
reco_model_1_pred <-  recosystem_model_1$predict(test_set_reco, out_memory())

#optimization_1
reco_model_2_pred <-  recosystem_model_2$predict(test_set_reco, out_memory())


# We will then find their rmse's
# optimization_1
set.seed(1, sample.kind = "Rounding")

reco_model_1_rmse <- RMSE(test_set$rating, reco_model_1_pred)
reco_model_1_rmse

# We get an rmse of 0.7902 

# We will save that into our table for models

rmse_model_results <- bind_rows(rmse_model_results,
                                tibble(method = "Recosystem Model 1",
                                       rmse = reco_model_1_rmse))


# optimization_2
set.seed(1, sample.kind = "Rounding")

reco_model_2_rmse <- RMSE(test_set$rating, reco_model_2_pred)
reco_model_2_rmse

# We get an rmse of 0.795

# We will save that into our table for models

rmse_model_results <- bind_rows(rmse_model_results,
                                tibble(method = "Recosystem Model 2",
                                       rmse = reco_model_2_rmse))


rmse_model_results %>% arrange(rmse)


# From the table we can see that from the 2 rmse's that we get with the recosystem models,
# the lowest is from the reco_model_1 and it's parameters

# Keep in mind that that we can also change the number of iteration with the argument niter so let's
# do that in the reco_model_1.
# However we have to be careful because if we add too many iterations we might over fit our model.
# Now we will attempt to increase the iterations to 35

# re-train the recosystem_model_1 with 35 iterations
recosystem_model_1$train(train_set_reco,
                         opts = c(optimization_1$min, niter = 35))
# Obtain the predictions

reco_model_1_pred <-  recosystem_model_1$predict(test_set_reco, out_memory())
# re-calculate the 

set.seed(1, sample.kind = "Rounding")
reco_model_1_rmse <- RMSE(test_set$rating, reco_model_1_pred)

# Add that to our table
rmse_model_results <- bind_rows(rmse_model_results,
                                tibble(method = "Recosystem Model 1 with 35 iteraations",
                                       rmse = reco_model_1_rmse))


# We can see a small improvement from the original reco_model_1 where it drops to 0.789. We will
# stop at 35 iterations which is a good number for the data set of our size.

# So we will create 2 different final models. One will be the "non-recosystem models" where we will
# use the regularized movie and user rating effects. While from the recosystem models we will use the first one.

###############################
# Final Model 1 and Model 2
###############################


###############################
# Final Model 1
###############################
# Now we will write our final model that will be tested by the the final holdout test

# average of the ratings
final_model_mu <- mean(edx$rating)

# the movie rating effect b_i
final_model_movie_reg_avg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - final_model_mu)/(n() +lambda), n_i =n())

# the user rating effect b_u
final_model_user_reg_avg <- edx %>%
  left_join(final_model_movie_reg_avg, by = 'movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = sum(rating - final_model_mu - b_i)/(n()+lambda), n_i = n())

# the regularized movie and user effect
final_model_user_reg_avg_pred <- final_holdout_test %>%
  left_join(final_model_movie_reg_avg, by = 'movieId') %>%
  left_join(final_model_user_reg_avg, by = "userId") %>%
  mutate(pred = final_model_mu + b_i + b_u) %>%
  pull(pred)

# The classic approach
set.seed(1, sample.kind = "Rounding")

# We get a final RMSE of 0.8648 on our first final model
FINAL_RMSE <- RMSE(final_holdout_test$rating, final_model_user_reg_avg_pred)
FINAL_RMSE



#########################################
# Final Model 2: Ricosystem Model
#########################################

# we set the seed to 1
set.seed(1, sample.kind = "Rounding")

# we make the final_holdout_test to the format that the recosystem library wants

final_model_reco_edx_data  <-  with(edx,  data_memory(user_index = userId, 
                                                      item_index = movieId, 
                                                      rating     = rating,
                                                      dates     = dates))
# we make the final_holdout_test to the format of the recosystem library
final_model_reco_val_data  <-  with(final_holdout_test,  data_memory(user_index = userId, 
                                                                     item_index = movieId, 
                                                                     rating     = rating,
                                                                     dates     = dates))

# We define our final recosystem model
final_model_recosystem <-  recosystem::Reco()

# We set the best preforming parameters to our final model
final_model_tuning_params <- final_model_recosystem$tune(final_model_reco_edx_data, 
                                                         opts = list(dim = c(10, 20, 30),          # dim is the number of latent factors.
                                                                     lrate = c(0.1, 0.2),          # lrate is the learning rate.
                                                                     costp_l2 = c(0.01, 0.1),      # costp_l2 is the L2 regularization cost for the user factors.
                                                                     costq_l2 = c(0.01, 0.1),      # costq_l2 is the L2 regularization cost for the item factors.
                                                                     nthread  = 4,                 # nthread is an integer and is the number of threads for parallel computing.
                                                                     niter = 10))                  # niter is an integer and is the number of iterations.


# We set the seed to 1
set.seed(1, sample.kind = "Rounding")

# We train the final model with the parameters
final_model_recosystem$train(final_model_reco_edx_data,
                             opts = c(final_model_tuning_params$min, niter = 35))
# We predict with the final recosystem model on the final_model_reco_val_data set that is the
# final_holdout_test but in the recosystem format
final_model_recosystem_pred <-  final_model_recosystem$predict(final_model_reco_val_data, out_memory())

# We set the seed to 1
set.seed(1, sample.kind = "Rounding")

# We obtain the final predictions of our RMSE
final_model_reco_rmse <- RMSE(final_holdout_test$rating, final_model_recosystem_pred)
final_model_reco_rmse


################################
# The end of the R script
################################