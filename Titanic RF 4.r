## Load packages
library(dplyr)
library(ggplot2)
library(Metrics)
library(randomForest)
library(party)
library(e1071)
library(caret)
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
str(combi)

## Summary
summary(combi)
str(combi)

## Basic features
combi$Age[is.na(combi$Age)] = -1
combi$Fare[is.na(combi$Fare)] = median(combi$Fare, na.rm=TRUE)
combi$Embarked[combi$Embarked==""] = "S"
combi$Sex = as.factor(combi$Sex)
combi$Embarked = as.factor(combi$Embarked)
combi$Survived = as.factor(combi$Survived)

## Engineered variable: Title
combi$Name = as.character(combi$Name)
combi$Title = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title = sub(' ', '', combi$Title)
combi$TitleOrig = combi$Title
table(combi$Title)
combi$Title[combi$Title %in% c('Capt', 'Col', 'Major', 'Dr')] = 'Officer'
combi$Title[combi$Title %in% c('Jonkheer', 'Don', 'Sir')] = 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess')] = 'Lady'
combi$Title[combi$Title %in% c('Mme')] = 'Mrs'
combi$Title[combi$Title %in% c('Mlle')] = 'Miss'
combi$Title[combi$Title %in% c('Ms')] = 'Mrs'
combi$Title = factor(combi$Title)
table(combi$Title)

## Engineered variable: Family size
combi = mutate(combi, FamilySize = SibSp + Parch + 1)

## Engineered variable: FamilyID
combi$Surname = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID = paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] = 'Small'
famIDs = data.frame(table(combi$FamilyID))
famIDs = famIDs[famIDs$Freq <= 2,] # Delete erroneous family IDs
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] = 'Small'
combi$FamilyID = factor(combi$FamilyID)
remove(famIDs)

## Engineered variable: FamilyID2
# Less factors for Random Forest
combi$FamilyID2 = combi$FamilyID
combi$FamilyID2 = as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] = 'Small'
combi$FamilyID2 = factor(combi$FamilyID2)

## Filling missing values: Age
plot(combi$Age)
plot(x = combi$Title, y = combi$Age)

for(t in levels(combi$Title)) {
  title_median = median(filter(combi, Age != -1 & Title == t)$Age)
  title_mean = mean(filter(combi, Age != -1 & Title == t)$Age)
  if(nrow(combi[combi$Title == t & combi$Age == -1,]) > 0)
    combi[combi$Title == t & combi$Age == -1,]$Age = title_median
}

remove(t, title_mean, title_median)

## Engineered variable: Child
combi$Child = 0
combi$Child[combi$Age <= 18 & combi$Parch > 0] = 1

## Engineered variable: Mother
combi$Mother = 0
combi$Mother[combi$Sex == 'female' & combi$Parch > 0 & combi$Age > 18 & combi$Title != 'Miss'] = 1

## Engineered variable: Deck from Cabin
cabin = as.character(combi$Cabin)
combi$Deck = sapply(cabin, function(x) strsplit(x,NULL)[[1]][1])
combi$Deck = as.factor(combi$Deck)

## Engineered variable: CabinPos
combi$CabinNum = sapply(cabin,function(x) strsplit(x,'[A-Z]')[[1]][2])
combi$num = as.numeric(combi$CabinNum)
num = combi$num[!is.na(combi$num)]
Pos = kmeans(num,3)
combi$CabinPos[!is.na(combi$num)] = Pos$cluster
combi$CabinPos = factor(combi$CabinPos)
levels(combi$CabinPos) = c('Front','End','Middle')
combi$num = NULL
remove(num, Pos, cabin)

## Inspect
str(combi)
summary(combi)

## Build model: Party RF
set.seed(415)
crf = cforest( Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2 + CabinPos + Deck + Mother + Child,
               data=combi[combi$Set == "Train",],
               controls=cforest_unbiased(ntree=2000, mtry=3)
              )
#party:::prettytree(crf@ensemble[[1]], names(crf@data@get("input")))
pred.crf = predict(crf, combi[combi$Set == "Test",], OOB=TRUE, type = "response") # Print tree
submission.crf = data.frame(PassengerId = combi[combi$Set == "Test",]$PassengerId, Survived = pred.crf)
write.csv(submission.crf, file = "4_random_forest_r_submission.csv", row.names=FALSE)


## Plots

plot(combi$Age, combi$FamilySize, col = ifelse(combi$Survived == 0,'red','green'), pch = 19 )

