---
title: "Practical machine learning  lectures 4"
author: "Kirill Setdekov"
date: "15 01 2020"
output:
  html_document:
    keep_md: yes
---



## Task:
* The goal of your project is to predict the manner in which they did the exercise. 
* This is the "classe" variable in the training set. You may use any of the other variables to predict with. 
* You should create a report describing how you built your model, 
* how you used cross validation, 
* what you think the expected out of sample error is, and 
* why you made the choices you did. 
* You will also use your prediction model to predict 20 different test cases.


## Data loading


```r
if(!file.exists("./data")) {
     dir.create("./data")
}
if (!file.exists("./data/training.csv") |
    !file.exists("./data/testing.csv")) {
     fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
     fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
     download.file(fileUrl1, destfile = "./data/training.csv")
     download.file(fileUrl2, destfile = "./data/testing.csv")
}

data <- read.csv("./data/training.csv")
quiz <- read.csv("./data/testing.csv")
```
## Train and test partition creation and NA handling

Some columns are full of NAs and are NA in the final 12 values, so I mark and remove them from the validation, and build dataset. Then i partition build dataset (original train csv) into train and test parts.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.6.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
## temp
shortenbuild <- createDataPartition(y=data$classe, p=0.1, list = FALSE)
data <- data[shortenbuild,]

## temp

allmissing <- sapply(quiz, function(x)!all(is.na(x)))
data[is.na(data)] <- 0
quiz[is.na(quiz)] <- 0
data <- data[,allmissing]
quiz <- quiz[,allmissing]

data <- data[,-1] 
quiz <- quiz[,-1] 


inBuild <- createDataPartition(y = data$classe,
                               p = 0.7, list = FALSE)
validation <- data[-inBuild, ]
buildData <- data[inBuild, ]

inTrain <- createDataPartition(y = buildData$classe,
                               p = 0.7, list = FALSE)
training <- buildData[inTrain, ]
testing <- buildData[-inTrain, ]
dim(training)
```

```
## [1] 967  59
```

```r
dim(testing)
```

```
## [1] 410  59
```

```r
dim(validation)
```

```
## [1] 587  59
```

### build 3 models 

I deliberately chose not to eliminate variables to test how different models perform on an unprocessed dataset.

I chose 4 methods: 
1. Generalized Boosted Model
2. Random forest
3. Linear discriminant analysis
4. Ensemble of the 1-3 above.


```r
set.seed(45)
mod1 <- train(classe~., method ="gbm", data = training, verbose = FALSE)
mod2 <- train(classe~., method ="rf", data = training, trControl = trainControl(method = "cv"),number = 3, verbose = FALSE)
mod3 <- train(classe~., method ="lda", data = training, verbose = FALSE)
```
### compare them

```r
pred1 <- predict(mod1, testing)
pred2 <- predict(mod2, testing)
pred3 <- predict(mod3, testing)
predDF3 <- data.frame(pred1, pred2, pred3, classe = testing$classe)

MLmetrics::Accuracy(pred1,testing$classe)
```

```
## [1] 0.9634146
```

```r
MLmetrics::Accuracy(pred2,testing$classe)
```

```
## [1] 0.9536585
```

```r
MLmetrics::Accuracy(pred3,testing$classe)
```

```
## [1] 0.8585366
```

### make a model that combines predictors


```r
combModFit3 <- train(classe~., method = "gbm", data = predDF3, verbose = FALSE)
combPred <- predict(combModFit3,predDF3)
MLmetrics::Accuracy(combPred,testing$classe)
```

```
## [1] 0.9829268
```

### on a validation

```r
pred1V <- predict(mod1, validation)
pred2V <- predict(mod2, validation)
pred3V <- predict(mod3, validation)

predVDF <-
    data.frame(
        pred1 = pred1V,
        pred2 = pred2V,
        pred3 = pred3V
    )
combPredV <- predict(combModFit3,predVDF)
MLmetrics::Accuracy(y_pred = pred1V,y_true = validation$classe)
```

```
## [1] 0.9608177
```

```r
MLmetrics::Accuracy(y_pred = pred2V,y_true = validation$classe)
```

```
## [1] 0.9574106
```

```r
MLmetrics::Accuracy(y_pred = pred3V,y_true = validation$classe)
```

```
## [1] 0.8517888
```

```r
## this is a combined model
MLmetrics::Accuracy(y_pred = combPredV,y_true = validation$classe)
```

```
## [1] 0.9744463
```

```r
confusionMatrix(combPredV,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 167   1   0   0   0
##          B   0 110   3   0   0
##          C   0   2  99   5   0
##          D   0   1   0  90   2
##          E   0   0   0   1 106
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9744          
##                  95% CI : (0.9582, 0.9856)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9677          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9649   0.9706   0.9375   0.9815
## Specificity            0.9976   0.9937   0.9856   0.9939   0.9979
## Pos Pred Value         0.9940   0.9735   0.9340   0.9677   0.9907
## Neg Pred Value         1.0000   0.9916   0.9938   0.9879   0.9958
## Prevalence             0.2845   0.1942   0.1738   0.1635   0.1840
## Detection Rate         0.2845   0.1874   0.1687   0.1533   0.1806
## Detection Prevalence   0.2862   0.1925   0.1806   0.1584   0.1823
## Balanced Accuracy      0.9988   0.9793   0.9781   0.9657   0.9897
```


