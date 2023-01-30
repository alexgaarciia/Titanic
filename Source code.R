# May we set the same seed for getting the same results:
set.seed(1)

#-------------------------------------------------------------------------------
#                                   PRE-STEPS:
#-------------------------------------------------------------------------------
# Load the data:
load('titanic_train.Rdata')
data = titanic.train

# str(data) for having a visual way of seeing the type of each variable in our
# dataset.
str(data)

# Libraries that will be used in the script:
if (!require("ggplot2")){
  install.packages('ggplot2')
}

if (!require("rpart")){
  install.packages('rpart')
}

if (!require("rpart.plot")){
  install.packages('rpart.plot')
}

if (!require("rattle")){
  install.packages('rattle')
}

if (!require("caTools")){
  install.packages('caTools')
}

if (!require("caret")){
  install.packages('caret')
}

if (!require("randomForest")){
  install.packages('randomForest')
}

if (!require("ggpubr")){
  install.packages('ggpubr')
}

if (!require("mlbench")){
  install.packages('mlbench')
}


#-------------------------------------------------------------------------------
#                                   ITEM 1:
#-------------------------------------------------------------------------------
# Before applying any machine learning algorithm, we have to make several tasks
# related to data, which could influence the over-fitting, under-fitting and
# validation process.Moreover, as the only machine learning techniques that 
# will be used along the whole project are decisions trees and random forests,
# these pre-processing steps are not necessary to be performed.
# For this part of the project, we will use classification trees. These are used
# when the data set needs to be split into classes that belong to the response
# variable. These are going to be really helpful when confirming or not the 
# conclusions that were done in the first assignment. 

# Note that the variables Cabin and Ticket do not contribute to the extraction
# of vital information from Survival. Hence, we will be working with the
# variables Age, Sex, Pclass, Sibsp,Parch, Fare and Embarked.

# General tree with valid variables.
general_tree = rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                     data=data, 
                     method="class")

fancyRpartPlot(general_tree,
               palettes = "BuPu",
               type = 2,
               digits=2,
               box.palette = "GnBu")

# Histogram for supporting a statement.
ggplot(titanic.train, aes(Age, fill = Survived)) + 
  geom_histogram() + 
  scale_fill_discrete(name = "Survived", labels= c("No", "Yes"))+
  facet_wrap(.~Sex) +
  ylab('')

# Trees for the conclusions taken in the first assignment.
tree_Pclass = rpart(Survived ~ Pclass,
                    data=titanic.train, 
                    method="class")
fancyRpartPlot(tree_Pclass,
               palettes = "BuPu",
               type = 2,
               digits=2,
               box.palette = "GnBu")

tree_Fare = rpart(Survived ~ Fare,
                  data=titanic.train, 
                  method="class")

fancyRpartPlot(tree_Fare,
               palettes = "BuPu",
               type = 2,
               digits=2,
               box.palette = "GnBu")


#-------------------------------------------------------------------------------
#                                    ITEM 2
#-------------------------------------------------------------------------------
data$Ticket = NULL
data$Cabin = NULL

# Once the data have been studied, the next step is to use a machine learning
# algorithm to create a model capable of representing the patterns present
# in the training data and generalizing them to new observations.
# Finding the best model is not easy, there are many algorithms, each with its
# own characteristics and with different parameters that must be adjusted. 
# We need to do the steps twice; one for decision trees and another one for
# random forests.
# First off, as indicated during the lectures, the very first thing we must do
# is choosing a sampling method. In our case, we chose the cross validation method.
# The goal of a predictive model is not to be able to predict observations that
# are already known, but new observations that the model has not seen. In order
# to estimate the error made by a model, it is necessary to resort to validation
# strategies, among which cross-validation stands out.


#-------------------------------------------------------------------------------
#                              DECISION TREES:
#-------------------------------------------------------------------------------
set.seed(1)
# For this specific way of sampling method, several steps must be performed.
# 1. Run the classification algorithm implemented in package r-part
# with default hyper-parameters values.
mytree=rpart(formula=data$Survived~., data=data, method="class")

# 2. Use the function predict to apply the
# classification algorithm to the current data set.
pred = predict(mytree,data,type="class")

# 3. Compute the confusion matrix. A confusion matrix in R is a table that
# will categorize the predictions against the actual values. It includes
# two dimensions, among them one will indicate the predicted values and
# another one will represent the actual values.
conf_matrix = table(data$Survived,pred,dnn=c("Actual value","Classifier prediction"))
conf_matrix_prop = prop.table(conf_matrix)

# 4. Choose a performance measure for comparing models. Compute error estimates.
accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
precision = conf_matrix[1,1]/sum(conf_matrix[,1])
specificity = conf_matrix[2,2]/sum(conf_matrix[,2])

# 5. K-fold cross validation using a lapply function.
folds = createFolds(data$Survived, k=10)
cv = lapply(folds, function(x) {
  # 1. Select training and test set according to current split
  training_set = data[-x,]
  test_set = data[x,]
  mytree=rpart(formula=Survived ~., data=training_set, method="class")
  
  # 2. Predict
  pred = predict(mytree,test_set,type="class")
  
  # 3. Confusion matrix
  conf_matrix = table(test_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))

  # 4. Compute error estimates
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix) 
  precision = conf_matrix[1,1]/sum(conf_matrix[,1])
  specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
  return(c(accuracy,precision,specificity))
})

# We un-list the list and get a vector and join the vectors in one single vector.
# We want to transform that vector so that we have a matrix. We use the transpose 
# for setting the performance measures in 3 columns.
cv = t(matrix(unlist(cv),nrow=3))

# We change the name of the columns.
colnames(cv)=c('Accuracy','Precision','Specificity')

# Compute the mean of the performance measures.
accuracies = apply(X=cv, MARGIN=2, FUN = "mean")


#-------------------------------------------------------------------------------
#                 HYPERPARAMETER SELECTION FOR DECISION TREEES:
#-------------------------------------------------------------------------------
# Once the sampling method has been successfully done, for each algorithm,
# we select the hyperparameters that optimize the performance measure. 
# Hence, We select the best combination of hyperparameters by trying several
# combinations and comparing the accuracy. We should make the average in
# precision, accuracy and specificity and select the best model.
# Many models contain parameters that cannot be learned from the training data
# and therefore must be set by the analyst. These are known as hyperparameters.
# The results of a model can depend to a great extent on the value that its
# hyperparameters take, however, it is not possible to know in advance which
# is the appropriate one. The most common way to find optimal values
# is by trying different possibilities.

# Hyperparameter selection
d_minsplit = seq(10,60,10)
d_cp=2^(-5:-11)
parameters=expand.grid(minsplit=d_minsplit,cp=d_cp)


folds = createFolds(data$Survived, k = 10)
cv_hyper = apply(parameters,1,function(y){
  cv = lapply(folds, function(x) {
    
    # 1. Select training and test set according to current split
    training_set = data[-x,]
    test_set = data[x,]
    mytree=rpart(formula=Survived ~., data=training_set, method="class",
                 control = rpart.control(minsplit=y[1],cp=y[2]))
    
    # 2. Use the function predict to apply the classification algorithm with test set
    pred = predict(mytree,test_set,type="class")
    
    # 3. Compute the confusion matrix
    conf_matrix = table(test_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))
    conf_matrix_prop = prop.table(conf_matrix)
    
    # 4. Compute error estimates
    accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
    precision = conf_matrix[1,1]/sum(conf_matrix[,1])
    specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
    return(c(accuracy,precision,specificity))
  })
  cv = t(matrix(unlist(cv),nrow=3))
  accuracies = apply(X=cv,MARGIN=2,FUN = "mean")  
  
  return(accuracies)
  
})

# We use the transpose for setting the performance measures in 3 columns.
cv_hyper1 = as.data.frame(t(cv_hyper))

# We change the name of the columns.
colnames(cv_hyper1) = c("Accuracy", "Precision", "Specificity")

parameters = cbind(parameters, cv_hyper1)

# Let's see how does the accuracy evolves.
plot(cv_hyper[1,])

# Now we are able to get the best model from the decision tree method.
cv_hyper[1,which.max(cv_hyper[1,])]
parameters[which.max(cv_hyper[1,]),]

# Visual evolution of the performance measures.
ggplot(parameters)+aes(x = cp, y = Accuracy, color = as.factor(minsplit))+
  geom_point(size = 1) + geom_line(size = 1)
ggplot(parameters)+aes(x = cp, y = Precision, color = as.factor(minsplit))+
  geom_point(size = 1) + geom_line(size = 1)
ggplot(parameters)+aes(x = cp, y = Specificity, color = as.factor(minsplit))+
  geom_point(size = 1) + geom_line(size = 1)


#-------------------------------------------------------------------------------
#                                RANDOM FORESTS:
#-------------------------------------------------------------------------------
# Run the classification algorithm implemented in
# package randomForest with default hyper-parameters values
classifier = randomForest(formula = Survived~.,
                          data = data,
                          ntree = 500)
plot(classifier)

# Choosing the number of trees and dividing the dataset.
folds = createFolds(data$Survived, k=10)
training_set = data[-folds[[1]],]
test_set = data[folds[[1]],]

# K-fold cross-validation
cv = lapply(folds,function(x){
  training_set = data[-x,]
  test_set = data[x,]
  classifier = randomForest(formula=Survived~.,
                            data = training_set,
                            ntree = 500)
  pred = predict(classifier, test_set)
  
  # Compute the confusion matrix
  conf_matrix = table(test_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))

  # Compute error estimates
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  precision = conf_matrix[1,1]/sum(conf_matrix[,1])
  specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
  return(c(accuracy,precision,specificity))
})

# We use the transpose for setting the performance measures in 3 columns.
cv = t(matrix(unlist(cv),nrow=3))

# We change the name of the columns.
colnames(cv)=c('Accuracy','Precision','Specificity')

# Compute the mean of the performance measures.
accuracies = apply(X=cv, MARGIN=2, FUN = "mean")


#-------------------------------------------------------------------------------
#                 HYPERPARAMETER SELECTION FOR RANDOM FORESTS:
#-------------------------------------------------------------------------------
d_mtry = seq(from=2,6,by=1)
d_ntree = seq(from=100,to=600,by=100)

# 'y' is the first combination of my first entry parameters.
parameters = expand.grid(mtry=d_mtry,ntree=d_ntree)
y = parameters[1,]

cv_hyper = apply(parameters, MARGIN = 1, function(y){cv = lapply(folds,function(x){
  training_set = data[-x,]
  test_set = data[x,]
  classifier = randomForest(formula=Survived~.,
                            data = training_set,
                            d_mtry=y[1],
                            d_ntree = y[2])
  pred = predict(classifier, test_set)
  
  # Compute the confusion matrix
  conf_matrix = table(test_set$Survived,pred,dnn=c("Actual value","Classifier prediction"))
  conf_matrix_prop = prop.table(conf_matrix)
  
  # Compute error estimates
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  precision = conf_matrix[1,1]/sum(conf_matrix[,1])
  specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
  return(c(accuracy,precision,specificity))
})
cv = t(matrix(unlist(cv), nrow = 3))
accuracies = apply(cv, MARGIN = 2, FUN = "mean")

return(accuracies)
})    

# We use the transpose for setting the performance measures in 3 columns.
cv_hyper2 = as.data.frame(t(cv_hyper))

# We change the name of the columns.
colnames(cv_hyper2) = c("Accuracy", "Precision", "Specificity")

parameters = cbind(parameters, cv_hyper2)

# Let's see how does the accuracy evolves.
plot(cv_hyper[1,])

# The row of parameters which are giving me the best hyper parameter combination.
posbest = which.max(cv_hyper[1,])

# We use parameters[posbest,] for knowing the rows of parameters that are
# giving me the best hyperparameter combination, with the highest accuracy.
parameters[posbest,]

bestclassifier = randomForest(formula=Survived~., data = data,
                             d_mtry = parameters[posbest,1],
                             d_ntree = parameters[posbest,2])

ggplot(parameters)+aes(x = mtry, y = Accuracy, color = as.factor(ntree))+
  geom_point(size = 1) + geom_line(size = 1)
ggplot(parameters)+aes(x = mtry, y = Precision, color = as.factor(ntree))+
  geom_point(size = 1) + geom_line(size = 1)
ggplot(parameters)+aes(x = mtry, y = Specificity, color = as.factor(ntree))+
  geom_point(size = 1) + geom_line(size = 1)


#-------------------------------------------------------------------------------
#                            FITTING THE BEST MODEL
#-------------------------------------------------------------------------------
# Lastly, we need a final function for checking the performance of the model.
my_model = function(test_set){
  test_set$Ticket = NULL
  test_set$Cabin = NULL
  pred = predict(bestclassifier, test_set, type = "class")
  conf_matrix = table(test_set$Survived, pred)
  accuracy = sum(diag(conf_matrix))/sum(conf_matrix)
  precision = conf_matrix[1,1]/sum(conf_matrix[,1])
  specificity = conf_matrix[2,2]/sum(conf_matrix[,2])
  return(list(prediction = pred,
              conf_matrix = conf_matrix,
              accuracy = accuracy,
              precision =precision,
              specificity = specificity))
}

save(bestclassifier, my_model,file='BestModel.RData')