# Set work direcetory
setwd("~/Documents/Data Science/Personal Project/urban_data_classification")

# Load library
library("tidyverse")
library("janitor")
library("tidymodels")

test_data <- read_csv("testing.csv")
train_data <- read_csv("training.csv")

glimpse(test_data)

sum(is.na(test_data))
sum(is.na(train_data))

nrow(train_data) # 168 observation
nrow(test_data) # 507

# EDA
test_data %>%
  count(class)

train_data %>% 
  count(class)

compare_df_cols_same(test_data, train_data)

urban_data <- bind_rows(train_data, test_data) %>% 
  mutate(id = row_number()) %>%
  select(id, everything()) %>% 
  clean_names() %>% 
  mutate_if(is.character, factor)

get_dupes(urban_data)

urban_data <- urban_data %>%
  select(-id)

write_csv(urban_data, "urban_data.csv")


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



# Set seed to ensure reproducible result
set.seed(120)

# Split the data to training and testing data
urban_data_split <- initial_split(urban_data,
                                  # set strata to compensate for class imbalance
                                  strata = class, 
                                  prop = 0.7)

urban_train <- training(urban_data_split)
urban_test <- testing(urban_data_split)

urban_train_resample <- vfold_cv(urban_train, v = 10)

urban_trurban_train_rec <-
  recipe(class ~., data = urban_train) %>% 
  step_normalize(all_numeric_predictors())

# Start work flow
knn_workflow <- workflow()

# Define nearest neighbour model
knn_model <- nearest_neighbor(neighbors = tune(),
                              dist_power = 2,
                              engine = "kknn",
                              mode = "classification")



# add model to workflow
knn_workflow <- knn_workflow %>%
  add_model(knn_model) 

knn_workflow <- knn_workflow %>% 
  add_recipe(urban_train_rec)


knn_grid <- tune_grid(
  knn_workflow,
  resamples = urban_train_resample
)

knn_grid %>%
  collect_metrics()

knn_grid %>% 
  show_best("accuracy")

knn_grid %>% 
  show_best("roc_auc")


# random_forest
rand_forest()


# fit the model
knn_fit <- fit_resamples(knn_workflow,
                         urban_train_resample)


# Resampling
vfold_cv(urban_train, v = 10, repeats = 10)

# split train
urban_train %>%
  count(class) %>%
  pull(n)

urban_test %>%
  count(class)%>%
  pull(n)


grid <- expand_grid(k = seq(2, 25, 1))

knn_model <- nearest_neighbor(
  neighbors = 10,
  dist_power = 2) %>%
  set_engine("kknn") %>%
  set_mode("classification")

urban_train_fit <- knn_model %>%
  fit(class~., data = urban_train)

urban_train_fit



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

#########################################################

rf_spec <- rand_forest(mode = "classification",
                       mtry = tune(),
                       trees = 1000,
                       engine = "randomForest",
                       min_n = tune())

tune_grid(
  class ~ .,
  model = rf_spec,
  resamples = 
)