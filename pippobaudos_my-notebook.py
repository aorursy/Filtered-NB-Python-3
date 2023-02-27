# Apply the Random Forest Algorithm
randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title,
                          data = train, importance = TRUE, ntree = 1000)

# Make your prediction using the test set
predict(my_forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)
