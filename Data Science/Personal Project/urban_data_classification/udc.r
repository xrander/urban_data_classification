# Load library
library("tidyverse") # for data manipulation and visualization
library("rsample") # for splitting data
library("class") # library for classification


# Read files from local disk
tst <- read_csv("/home/xrander/Documents/Data Science/Data/urban+land+cover/testing.csv")
trn <- read_csv("/home/xrander/Documents/Data Science/Data/urban+land+cover/training.csv")

# Data Understanding

ncol(tst) # number of variables for test data
ncol(trn) # number of variables for training data

nrow(tst) # num of obsersvations for test data
nrow(trn) # num of observations for training data

# Missing data
sum(is.na(tst))
sum(is.na(trn))

# Summarize data
summary(trn)
summary(tst)

# combine both data to a whole
urban_data <- bind_rows(trn, tst)

summary(urban_data)

# Frequency of the data
plot(table(urban_data$class),
     type = "h",
     main = "Frequency of Urban Class",
     xlab = "class",
     ylab = "frequency",
     col = "red")


# normalize the numerical variable in the data
urban_data_processed <- urban_data %>%
  mutate(class = factor(class,
                        labels = c("car", "concrete", "tree", "building",
                                   "asphalt", "grass","shadow", "soil","pool"),
                        levels = c("car", "concrete", "tree", "building",
                                   "asphalt", "grass","shadow", "soil","pool")),
         across(where(is.double), scale))

# Set seed to ensure reproducible result
set.seed(120)

# Split data
split <- initial_split(urban_data_processed, prop = 0.8)
training_data <- training(split)
testing_data <- testing(split)



training_data
training_dt_lab <- training_data$class

# Deciding the number of k
k <- ceiling(sqrt(nrow(training_data)))

# using knn
urban_class_pred <- knn(train = training_data[, -1], # select all rows except the label
                       test = testing_data[,-1],
                       cl = training_dt_lab,
                       k = 5) # select only the label row


table(prediction = urban_class_pred, actual = testing_data$class)
mean(urban_class_pred == testing_data$class)

library(gmodels)

CrossTable(x = testing_data$class, y = urban_class_pred,
           prop.chisq = F)

library(xgboost)

