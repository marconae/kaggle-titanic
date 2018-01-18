## Load packages
library(needs)
needs(dplyr, tidyr, stringr, lubridate, readr, ggplot2, MASS, pander, formattable, viridis, Metrics, randomForest)

## Files
list.files("./input") %>%
  as.character()

## Read
train = read.csv("./input/train.csv", stringsAsFactors = TRUE)
train$Set = "Train"
test = read.csv("./input/test.csv", stringsAsFactors = TRUE)
test$Survived = 0
test$Set = "Test"
combi = rbind(train, test)
remove(test)
remove(train)

## Basic Features
extractFeatures = function(data) {
  features = c("Pclass", "Age", "Sex", "Parch", "SibSp", "Fare", "Embarked", "Survived")

  fea = data[,features]
  fea$Age[is.na(fea$Age)] = -1
  fea$Fare[is.na(fea$Fare)] = median(fea$Fare, na.rm=TRUE)
  fea$Embarked[fea$Embarked==""] = "S"
  fea$Sex = as.factor(fea$Sex)
  fea$Embarked = as.factor(fea$Embarked)
  fea$Survived = as.factor(fea$Survived)
  return(fea)
}

## Sets for algorithm
data = extractFeatures(combi)
str(data)

## Feature Engineering: Title
data$Title = sapply(as.character(combi$Name), FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
data$Title <- sub(' ', '', data$Title)
table(data$Title)
data$Title[data$Title %in% c('Mme', 'Mlle')] = 'Mlle'
data$Title[data$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
data$Title[data$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'
table(data$Title)
data$Title <- factor(data$Title)

## Build model
rf = randomForest(Survived~., data=data[combi$Set == "Train",], ntree=100, importance=TRUE)
pred.rf = predict(rf, newdata = data[combi$Set == "Test",])

## Feature Importance
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

## Plot: Feature Importance
ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("") +
  ylab("Importance") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

## Write submission file
submission <- data.frame(PassengerId = test$PassengerId, Survived = pred.rf)
write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)
