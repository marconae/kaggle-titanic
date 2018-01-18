## Load packages
library(caTools)

## Read
train = read.csv("./input/train.csv", stringsAsFactors = FALSE)
test = read.csv("./input/test.csv", stringsAsFactors = FALSE)
train$Set = "Train"
test$Survived = 0
test$Set = "Test"
combi = rbind(train, test)
combi$Set = as.factor(combi$Set)
remove(test)
remove(train)

## Basic features
combi$Age[is.na(combi$Age)] = mean(combi$Age[!is.na(combi$Age)])
combi$Fare[is.na(combi$Fare)] = median(combi$Fare, na.rm=TRUE)
combi$Embarked[combi$Embarked==""] = "S"
combi$Sex = as.factor(combi$Sex)
combi$Embarked = as.factor(combi$Embarked)
combi$Survived = as.factor(combi$Survived)

## Summary
summary(combi)
str(combi)

## Simple model
split = sample.split(combi[combi$Set == "Train",]$Survived, SplitRatio = 0.7)
train = subset(combi[combi$Set == "Train",], split == T)
test = subset(combi[combi$Set == "Train",], split == F)
train$Set = NULL
test$Set = NULL

## Simple decision tree
library(rpart)

mod1 = rpart(Survived ~ Age + Sex, data=train)
pred1 = predict(mod1, newdata=test, type="class")

table(test$Survived, pred1)
