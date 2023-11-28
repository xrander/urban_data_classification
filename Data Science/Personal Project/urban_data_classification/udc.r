# Set work direcetory

#setwd("~/Documents/Data Science/Personal Project/urban_data_classification")

# Load library
library("tidyverse") # for data manipulation and visualization
library("tidymodels") # for splitting dat
library("class") # library for classification
library("caret") # for hyperparameter tuning

# set work directory
setwd("~/Documents/Data Science/Personal Project/urban_data_classification/")

# Read files from local disk
tst <- read_csv("testing.csv") # test data
trn <- read_csv("training.csv") # training data

# Missing data
sum(is.na(tst))
sum(is.na(trn)) # No missing data in the data frames

# get the number of observation
nrow(trn) # 168 observation
nrow(tst) # 507

# EDA
tst %>%
  count(class)# There are 9 classes

trn %>% 
  count(class)

# combine both data to a whole
# First we check if they have the same number and names of variables

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
# The classes are not equal

# normalize the numerical variable in the data
urban_data <- urban_data %>%
  mutate(class = factor(class,
                        labels = c("car", "concrete", "tree", "building",
                                   "asphalt", "grass","shadow", "soil","pool"),
                        levels = c("car", "concrete", "tree", "building",
                                   "asphalt", "grass","shadow", "soil","pool"),
                        ordered = F), # change class to factor
         across(where(is.double), scale))# scale the predictor

# Set seed to ensure reproducible result
set.seed(120)

# Split the data to training and testing data
urban_data_split <- initial_split(urban_data,
                                  # set strata to compensate for class imbalance
                                  strata = class, 
                                  prop = 0.7)

urban_train <- training(urban_data_split)
urban_test <- testing(urban_data_split)

# Start work flow
knn_workflow <- workflow()

# Define nearest neighbour model
knn_model <- nearest_neighbor(
  neighbors = 10,
  dist_power = 2) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# add model to workflow
knn_workflow <- knn_workflow %>%
  add_model(knn_model)

# add target and feature variables
knn_workflow <- knn_workflow %>%
  add_variables(outcomes = class,
                predictors = everything())

# preview workflow
knn_workflow

# fit the model
knn_fit <- fit(knn_workflow, urban_train)


# Resampling
vfold_cv(urban_train, v = 10, repeats = 10)

# split train
urban_train %>%
  count(class) %>%
  pull(n)

urban_test %>%
  count(class)%>%
  pull(n)

# split each class in the data
#urban_data_split<- split(urban_data, urban_data$class)

# make rsplit data for each class

#for (name in names(urban_data_split)){
#  assign(paste0(name, "_rsplit"), initial_split(urban_data_split[[name]], prop = 0.8))
#}


# Decide k using hyperparameter tuning result
#control <- trainControl(method = "cv", number = 10)

# square root of num of observation in training data
#round(sqrt(nrow(training_data)), 0) # to be included as part of k-values to choose from

# Create k values to choose from in a grid
grid <- expand_grid(k = seq(2, 25, 1))

knn_model <- nearest_neighbor(
  neighbors = 10,
  dist_power = 2) %>%
  set_engine("kknn") %>%
  set_mode("classification")

urban_train_fit <- knn_model %>%
  fit(class~., data = urban_train)

urban_train_fit

knn_value <- train(
  class ~ .,
  data = urban_train,
  method = "knn",
  tuneGrid = grid,
  trControl = trainControl(method = "cv", number = 10))

print(knn_value) #k = 13

k = knn_value$bestTune$k


# using knn to predict
urban_class_pred <- knn(train = urban_train[, -1], # select all rows except the class variable
                       test = urban_test[,-1],
                       cl = as.matrix(urban_train[, 1]), # select the class variable
                       k = k) # use best k value from hypertuning


# Confusion Matrix
table(prediction = urban_class_pred, actual = urban_test$class)

# Accuracy
mean(urban_class_pred == urban_test$class)

tibble("x" = testing_data$BrdIndx,
       "y" = testing_data$Round,
       "actual" = testing_data$class,
       "prediction" = urban_class_pred) %>%
  pivot_longer(cols = c("actual", "prediction"),
               names_to = "type",
               values_to = "value",
               values_drop_na = T) %>%
  filter(value == c("grass", "concrete", "tree", "asphalt", "soil", "building")) %>%
  ggplot(aes(x, y, color = value))+
  geom_point(size = 2)+
  geom_smooth(method = lm, se = F, alpha = 0.4)+
  facet_wrap(~type)
  


library(gmodels)

CrossTable(x = testing_data$class, y = urban_class_pred,
           prop.chisq = F)

# Load the xgboost library
library(xgboost)
# Creat xgboost data structure for training and test data
xgb_trn_data <- xgb.DMatrix(data = as.matrix(training_data[, -1]), label = as.matrix(training_data$class)) 


xgb.DMatrix(data = as.matrix(testing_data[, -1]))


