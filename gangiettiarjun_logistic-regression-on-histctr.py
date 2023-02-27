# Kaggle-competititon "Avito Context Ad Clicks"
# See https://www.kaggle.com/c/avito-context-ad-clicks

# In order to run this script on Kaggle-scripts I had to limit the number of entries to read
# from the database as well as to decrease the sample-size. With the full dataset from the database as well
# as a sample of 10 millions entries I got a 0.05104 on the public leaderboard

library("data.table")
library("RSQLite")
library("caret")

# ----- Prepare database -------------------------------------------------------


# ----- Utitlies ---------------------------------------------------------------

# Define constants to improve readability of large number

# Runs the query, fetches the given number of entries and returns a
# data.table

# Loss-function to evaluate result
# See https://www.kaggle.com/c/avito-context-ad-clicks/details/evaluation

# ----- Simple Machine Learning ------------------------------------------------

# Select contextual Ads (OnjectType=3), results in 190.157.735 entries
# Warning: Takes a few minutes
trainSearchStreamContextual <- fetch(db, "select HistCTR, IsClick from trainSearchStream where ObjectType=3", 10 * million)

# Create stratified sample 

trainSearchStreamContextualSample <- trainSearchStreamContextual[as.vector(sampleIndex), ]

# Compare click-ratio in full set and sample to verify stratification


# Create stratified random split ...


# ... and partition data-set into train- and validation-set
trainSearchStreamContextualTrainSample <- trainSearchStreamContextualSample[as.vector(trainSampleIndex),]
trainSearchStreamContextualValidationSample <- trainSearchStreamContextualSample[-as.vector(trainSampleIndex),]

# Build a logistic regression ...
model <- glm(IsClick ~ HistCTR, data = trainSearchStreamContextualTrainSample, family="binomial")

# Check that regression-coefficients have significant impact
summary(model)

# ... and predict data on validation data-set
prediction <- predict(model, trainSearchStreamContextualValidationSample, type="response")


# ----- Predict submission dataset ---------------------------------------------

testSearchStreamContextual <- fetch(db, "select TestId, HistCTR from testSearchStream where ObjectType=3")
prediction <- predict(model, testSearchStreamContextual, type="response")


submissionFile <- paste0("glm", format(Sys.time(), "%Y-%m-%d-%H:%M:%S"), ".csv")
#write.csv(submissionData, submissionFile, sep=",", dec=".", col.names=TRUE, row.names=FALSE)

# ----- Clean up ---------------------------------------------------------------

dbDisconnect(db)


