## Load packages
library(dplyr)
library(ggplot2)
library(Metrics)
library(randomForest)
library(party)
library(e1071)
library(caret)

## Read
train = read.csv("./input/train.csv", stringsAsFactors = TRUE)
train$Set = "Train"
test = read.csv("./input/test.csv", stringsAsFactors = TRUE)
test$Survived = 0
test$Set = "Test"
combi = rbind(train, test)
remove(test)
remove(train)

## Basic features
combi$Age[is.na(combi$Age)] = -1
combi$Fare[is.na(combi$Fare)] = median(combi$Fare, na.rm=TRUE)
combi$Embarked[combi$Embarked==""] = "S"
combi$Sex = as.factor(combi$Sex)
combi$Embarked = as.factor(combi$Embarked)
combi$Survived = as.factor(combi$Survived)

## Engineered variable: Child
combi$Child = 0
combi$Child[combi$Age < 18 & combi$Age != -1] = 1

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

# Engineered variable: FamilyID
combi$Surname = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID = paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] = 'Small'
famIDs = data.frame(table(combi$FamilyID))
famIDs = famIDs[famIDs$Freq <= 2,] # Delete erroneous family IDs
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] = 'Small'
combi$FamilyID = factor(combi$FamilyID)
remove(famIDs)

# Engineered variable: FamilyID2
# Less factors
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

## Inspect
str(combi)
summary(combi)

## Build model: standard RF
set.seed(415)
rf = randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2, 
                  data=combi[combi$Set == "Train",], 
                  ntree=2000, 
                  importance=TRUE
                  )

pred.rf = predict(rf, newdata = combi[combi$Set == "Test",])

## Feature Importance
imp = importance(rf, type=1)
featureImportance = data.frame(Feature=row.names(imp), Importance=imp[,1])

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
submission = data.frame(PassengerId = combi[combi$Set == "Test",]$PassengerId, Survived = pred.rf)
write.csv(submission, file = "3_random_forest_r_submission.csv", row.names=FALSE)

## Build model: Party RF
set.seed(415)
crf = cforest( Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data=combi[combi$Set == "Train",],
               controls=cforest_unbiased(ntree=2000, mtry=3)
              )
party:::prettytree(crf@ensemble[[1]], names(crf@data@get("input")))
pred.crf = predict(crf, combi[combi$Set == "Test",], OOB=TRUE, type = "response") # Print tree
submission.crf = data.frame(PassengerId = combi[combi$Set == "Test",]$PassengerId, Survived = pred.crf)
write.csv(submission.crf, file = "4_random_forest_r_submission.csv", row.names=FALSE)

## CARET
library(doMC)
registerDoMC(cores = 4)
rf.caret = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
                 data=combi[combi$Set == "Train",],
                 method="rf",
                 ntree=1000,
                 trControl=trainControl(method="cv",number=5),
                 prox=TRUE,
                 allowParallel=TRUE)
print(rf.caret)
