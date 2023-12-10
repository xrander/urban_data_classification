# Set work direcetory
setwd("~/Documents/Data Science/Personal Project/urban_data_classification")

# Load library
library("tidyverse")
library("janitor")
library("tidymodels")
library(vip)

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


skimr::skim(urban_data)

# Set seed to ensure reproducible result
set.seed(120)

# Split the data to training and testing data
urban_data_split <- initial_split(urban_data,
                                  # set strata to compensate for class imbalance
                                  strata = class, 
                                  prop = 0.7)

urban_train <- training(urban_data_split)
urban_test <- testing(urban_data_split)


urban_train_rec <-
  recipe(class ~., data = urban_train) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_zv(all_predictors())


urban_train_prep <- prep(urban_train_rec)

urban_train_prep

urban_train_juiced <- juice(urban_train_prep)

skimr::skim(urban_train_juiced)

# Define nearest neighbour model
knn_model <- nearest_neighbor(neighbors = tune(),
                              dist_power = 2,
                              engine = "kknn",
                              mode = "classification")

# Start work flow
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(urban_train_rec)


set.seed(2344)

urban_train_resample <- vfold_cv(urban_train, v = 10)

set.seed(2333)
doParallel::registerDoParallel()

# hyperparameter tuning
knn_tune <- tune_grid(
  knn_workflow,
  resamples = urban_train_resample
)

knn_tune %>%
  collect_metrics()

knn_tune %>% 
  show_best("accuracy")

knn_tune %>% 
  show_best("roc_auc")


knn_tune %>% 
  collect_metrics() %>% 
  select(neighbors, .metric, mean) %>% 
  clean_names() %>% 
  ggplot(aes(neighbors, mean, col = metric))+
  geom_point()+
  geom_line()+
  facet_wrap(~metric, scales = "free")

knn_grid_search <- grid_regular(
  neighbors(range = c(10,14)),
  levels = 5
  )

knn_tune_grid <- tune_grid(
  knn_workflow,
  resamples = urban_train_resample,
  grid = knn_grid_search
)

autoplot(knn_tune_grid, metric = "roc_auc")+
  labs(x = "Nearest Neighbor",
       y = "roc_auc",
       title = "Area under curve based on Nearest neighbor")
  

knn_best_auc <- knn_tune_grid %>% 
  select_best("roc_auc")

knn_final_model <- 
  finalize_model(
    knn_model,
    knn_best_auc
    )

knn_final_wf <- workflow() %>% 
  add_recipe(urban_train_rec) %>% 
  add_model(knn_final_model) 

knn_final_res <- knn_final_wf %>% 
  last_fit(urban_data_split)

knn_final_res %>% 
  collect_predictions() %>% 
  mutate(prediction = ifelse(class ==.pred_class, "correct", "wrong")) %>% 
  bind_cols(urban_test) %>% 
  ggplot(aes(ndvi, mean_g, col = prediction))+
  geom_point(alpha = 0.7)

  

knn_predict <- knn_final_res %>% 
  collect_predictions() %>% 
  clean_names()

conf_mat(knn_predict, truth = class, estimate = pred_class)

sens(knn_predict, truth = class, estimate = pred_class)
accuracy(knn_predict, truth = class, estimate = pred_class)

######################################################### # Random Forest

rf_model <- rand_forest(mode = "classification",
                       mtry = tune(),
                       trees = 1000,
                       engine = "ranger",
                       min_n = tune())

rf_workflow <- workflow() %>% 
  add_variables(outcomes = class,
                predictors = everything()) %>% 
  add_model(rf_model)
  
set.seed(234)

doParallel::registerDoParallel()

rf_tune <- tune_grid(
  rf_workflow,
  resamples = urban_train_resample,
  grid = 20
)

rf_tune %>% 
  collect_metrics()

rf_tune %>% 
  show_best("roc_auc")

rf_tune %>% 
  collect_metrics() %>% 
  filter(.metric == "roc_auc") %>% 
  select(mtry, min_n, mean) %>% 
  pivot_longer(mtry:min_n,
               names_to ="parameter",
               values_to = "values") %>% 
  ggplot(aes(values, mean, col = parameter))+
  geom_point()+
  facet_wrap(~parameter, scales = "free_x")


set.seed(345)

rf_grid <- grid_regular(
  mtry(range = c(130, 147)),
  min_n(range = c(4, 10)),
  levels = 5
  )


rf_tune_grid <- tune_grid(rf_workflow,
                            resamples = urban_train_resample,
                            grid = rf_grid
                            )

rf_tune_grid %>% 
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>% 
  select(mtry, min_n, mean) %>% 
  mutate(min_n = factor(min_n)) %>% 
  ggplot(aes(mtry, mean, col = min_n))+
  geom_point()+
  geom_line()

rf_tune_grid %>% 
  show_best("roc_auc")

rf_tune_grid %>% 
  show_best("accuracy")

best_tune_auc <- rf_tune_grid %>% 
  select_best("roc_auc")

rf_final_model <-
  finalize_model(
    rf_model,
    best_tune_auc
  )


rf_final_model %>% 
  set_engine("ranger", importance ="permutation") %>% 
  fit(class ~ .,
      data = urban_train) %>% 
  vip(geom = "point")


final_wf <- workflow() %>% 
  add_variables(outcomes = class,
                predictors = everything()) %>% 
  add_model(rf_final_model)

final_rf_res <- final_wf %>% 
  last_fit(urban_data_split)

final_rf_res %>% 
  collect_predictions() %>% 
  mutate(prediction= if_else(class == .pred_class, "correct", "wrong")) %>% 
  bind_cols(urban_test) %>% 
  ggplot(aes(ndvi,mean_g, color =prediction))+
  geom_point(alpha = 0.7)

rf_predict <- final_rf_res %>%
  collect_predictions() %>% 
  clean_names()

conf_mat(rf_predict, truth = class, estimate = pred_class)
sens(rf_predict, truth = class, estimate = pred_class)
