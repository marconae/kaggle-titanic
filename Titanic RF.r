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

## The Gender Class model
prop.table(table(train$Survived))
test$Survived <- rep(0, 418)
submit1 = data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit1, file = "theyallperish.csv", row.names = FALSE)

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

## Build model
rf = randomForest(Survived~., data=data[combi$Set == "Train",], ntree=100, importance=TRUE)
pred.rf = predict(rf, newdata = data[combi$Set == "Test",])

## Features
imp <- importance(rf, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("") +
  ylab("Importance") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

## Write submission file
submission <- data.frame(PassengerId = combi[combi$Set == "Test",]$PassengerId, Survived = pred.rf)
write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)
