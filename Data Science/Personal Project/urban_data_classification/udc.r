# Set work direcetory

#setwd("~/Documents/Data Science/Personal Project/urban_data_classification")

# Load library
library("tidyverse") # for data manipulation and visualization
library("rsample") # for splitting data
library("class") # library for classification
library("caret") # for hyperparameter tuning


# Read files from local disk
tst <- read_csv("testing.csv") # test data
trn <- read_csv("training.csv") # training data

# Missing data
sum(is.na(tst))
sum(is.na(trn))


# Summarize data
summary(trn)
summary(tst)

# combine both data to a whole
# First we chaeck if they have the same number and names of variables
ncol(trn) == ncol(tst) # Test for equal number of variables

unique(names(trn) ==  names(tst)) # Test for similar variables names

urban_data <- bind_rows(trn, tst) # combine the data

nrow(urban_data) == sum(nrow(trn), nrow(tst)) # check if observations are properly combined

write_csv(urban_data, "urban_data.csv")

# Target variable - class
# Explanatory variable - Everything else

# Distribution of the target variable
urban_data %>%
  group_by(class) %>% # group b
  summarize(frequency = n()) %>%
  ggplot(aes(class, frequency))+
  geom_bar(stat = "identity",
           fill = "burlywood3")+
  theme_bw()+
  ggtitle("Frequency of Classes")+
  geom_text(aes(label = frequency,
                vjust = 0.001))


# normalize the numerical variable in the data
urban_data <- urban_data %>%
  mutate(class = factor(class,
                        labels = c("car", "concrete", "tree", "building",
                                   "asphalt", "grass","shadow", "soil","pool"),
                        levels = c("car", "concrete", "tree", "building",
                                   "asphalt", "grass","shadow", "soil","pool")), # change class to factor
         across(where(is.double), scale))# scale the predictor


# Set seed to ensure reproducible result
set.seed(120)

# Split data
split <- initial_split(urban_data, prop = 0.8)
training_data <- training(split)
testing_data <- testing(split)


# Decide k using hyperparameter tuning result
control <- trainControl(method = "cv", number = 10)

# square root of num of observation in training data
round(sqrt(nrow(training_data)), 0) # to be included as part of k-values to choose from

# values to choose K from
k_values <- seq(1, 25, 2)

# Define tuning grid
grid <- expand.grid(k = k_values)

knn_value <- train(
  class ~ .,
  data = training_data,
  method = "knn",
  tuneGrid = grid,
  trControl = control)

print(knn_value) #k = 13

k = knn_value$finalModel$k

# using knn to predict
urban_class_pred <- knn(train = training_data[, -1], # select all rows except the class variable
                       test = testing_data[,-1],
                       cl = as.matrix(training_data[, 1]), # select the class variable
                       k = k) # use best k value from hypertuning


# Confusion Matrix
table(prediction = urban_class_pred, actual = testing_data$class)

# Accuracy
mean(urban_class_pred == testing_data$class)

tibble("actual" = testing_data$class,
       "prediction" = urban_class_pred)

library(gmodels)

CrossTable(x = testing_data$class, y = urban_class_pred,
           prop.chisq = F)

# Load the xgboost library
library(xgboost)
# Creat xgboost data structure for training and test data
xgb_trn_data <- xgb.DMatrix(data = as.matrix(training_data[, -1]), label = as.matrix(training_data$class)) 


xgb.DMatrix(data = as.matrix(testing_data[, -1]))