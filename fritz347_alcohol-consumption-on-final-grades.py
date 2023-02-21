#!/usr/bin/env python
# coding: utf-8



The primary goal of our analytics is discover if alcohol consumption has any predictive power or force 
    over student’s final grades in school. We are also interested to see how other variables compare 
    to alcohol consumption and whether alcohol consumption is the best predictor when forecasting a 
    student’s final grade. The business problem is relevant for any academy or university, family and 
        community  members, and the students. They can use this information to identify which students
        are at more risk to fail. With that information they can take the proper precautionary steps 
        to ensure that the particular student receives the help that he or she needs. A decrease in 
        failure rates increase student attendance(revenue) and the overall perception of the institute 
        itself. Families can use our analysis to better inform their children of the consequences that
        alcohol consumption has on grades. Students will be able to use our findings to become more 
        aware of the factors that affect their grades in school. We believe that the application of 
        our analysis has an incredibly large scale, and multiple parties will be very interested. 

Data Understanding
When we reviewed the strengths and limitations of our dataset we found a few things. First, there are 
    many variables, 30 to be exact, that help describe each student. These variables include information 
    on the students’ families, personal lives, and academics. We found that having all these variables 
    gave us a lot of options when we were exploring our data. Another strength of our data is there is
    no missing values. Each instance’s variables are complete with no null values, making our data very 
    consistent. A limitation we noticed right away was how small our sampling size was. There are only 
    395 total instances in our data set. Being that this sample is so small and taken from Portugal, we
        realized it is not the best representation of the entire population of students. Another 
        weakness we noticed was how the source assigned values to some of the attributes. We found that
        some can be limited. For example, the options for father’s job are only teacher, health care 
            related, civil services, at home or other. We found that the attributes that contain these 
            limitations can be somewhat skewed. Nonetheless, we were still able to yield interesting 
            results.The students are from two Portuguese secondary schools. Each gender is represented
            and the ages of the students range from 15 to 22. There are multiple demographic variables
            included in our dataset to describe each student. Variables include information about the 
            student’s family. For example, the family size, parent’s cohabitation status, each parent’s
            educational background and job, and the student’s relationship with their family. All these
            variables are categorical; however, they are receiving a score that represents their status.
        We are going to look to see if any of these demographic variables serve as better predictors of
            student’s final grades than alcohol consumption. Our dataset also includes variables that
            reveal information on the individual students. These variables include reason they chose 
            the school, extra-curricular activities, the amount of free time each student has, whether
            the student goes out or not, and current health status. Again, these variables are 
            categorical. We will be looking to see if these informational variables will provide us any
            helpful insight to solve our business problem. Furthermore, our dataset includes variables 
            that provide information on each of the student’s academics. These variables include 
            weekly study time, number of classes absence, number of previous classes failed, and 
            whether they want to pursue a higher education or not. Again, we will be looking to see
            if any of these variables will serve as better predictors of a student’s final grade rather
            than alcohol consumption. Lastly, there are two variables that define the student’s weekly 
            alcohol consumption. These two variables are workday alcohol consumption and weekend alcohol
                consumption. We are hoping that these two variables will be the strongest predictors of 
                whether a student passes or fails their course. In conclusion, we believe that we have 
                found a quality data set. The source is credible and the data set has been published.
                Combine that with our preliminary findings, we believe the data set to be accurate.

Data Preparation
First, the data did not include a class variable, so we created one. Our class variable is whether 
                the student passed their Math course. Our data set did include three grades: First 
                period grade, Second period Grade, and Final Grade. We eliminated the first two grade
                portions and kept the final because we are focusing on if their final grade was a pass
                or fail. We were not concerned with the midterm grades when doing our analytics.
                Next, we had to determine what defined a passing grade. The grading scale in Europe 
                   is different than the one the United States uses.  Portugal’s grading system is 
                   based on a scale of 1-20, and any score above a 9 is a passing grade. 
                   (16-20 A, 13-15 B, 10-12 C, Else F). That is how we decided on our class variable.
                   If a student received a 9 or above for their final grade (G3), they would receive 
                   a Pass, any score equal or less than 9 is considered a Fail. We continued by
                   observing the types of variables in our data set. We began to question if we 
                   needed to transform any of the data. This next step included changing the numeric
                   variables into categorical to help us more easily understand the type of information
                   the variable was trying to express. To indicate for instance, “Very bad”, “Bad”, 
                   “Okay”, “Good”, “Very Good”, is represented by a scale of 1-5.  Since nearly 80% of 
                   our variables were categorical variables represented by numeric values we were 
                   anticipating spending a lot of time having to transform them into categorical.
                   Fortunately, all the variables we were preparing to change were already 
                   identified as factors. However, we did need to transform our class variable, 
                       Pass/Fail, into a factor. We were able to do that very easily using RStudio.
After transforming some of our data, we attempted to further clean up our data set. First, we looked
                       to see if any data cleaning would be necessary. Luckily, our data set was not
                           noisy. We did not find any inconsistencies, and we did not have to handle 
                           any missing data. Second, we attempted to prepare our data by reducing the
                           size of it. We did not eliminate any instances, but we did eradicate some 
                           variables. We noticed that some of the variables were not going to help us
                           solve the problem we were hoping to address. Immediately we ran a simple 
                           information gain in Weka to see which variables are the most informative
                           and which are the least. Most of the variables we removed contained 
                           demographic information on an individual student. They gave us great 
                           insight on the students we were analyzing, but did not provide us with 
                           great feedback regarding our business goal. Despite removing those variables,
                           the accuracy of our analytics did not increase as we predicted 
                           it would. 


Descriptive Analytics
 
We started the descriptive analytics by looking at the attributes that provided the most information
gain. We did this by running the InfoGainAttributeEval in Weka. Failures,Go out, and Mother’s job were
among the top three.The Y-axis represents if a student has passed or failed the course. The X-axis 
represents how many classes how many previous classes the student has failed. In this sample, if the
student failed more than two previous classes, they almost always failed the Math course.
Since we want to focus on the effects of alcohol on the class variable we graphed average workday 
alcohol consumption and the class variable. Students who failed the class have a slightly higher 
workday alcohol intake than those who passed.This graph is average weekend alcohol consumption on class.
It is a very small difference but students who failed that class have a higher average weekend
alcohol consumption than those who passed. This graph is showing past failures on students who were
enrolled in the Math course. This is the attribute with the biggest information gain and the graph makes
it very easy to see why. Students who failed the Math course have an average of .6923 course failures 
 already. People who are passed the Math course only have an average of .1585 past failures. We believe
    the we would have more information gain if the values of past failures accurately displayed number
        of failures instead of displaying “n if 1<=n<3, else 4.” Some students may have failed more than
     4 classes but due to data limitations we are unable to accurately display that.
The second histogram depicts the average amount of going out compared to the class variable. We can see that students who failed the class went out roughly an average of .5 more than ones who passed. The students who failed went out in between “medium and high” on our scale but the students who passed only went out a “medium” amount.
 These tables show the 3rd most informative attribute which is wanting to take on higher education. Looking at the percent total table on the right we can see that kids who don’t want to take higher education failed the class 65%. There are way more kids who want to take on higher education than kids who don’t so this attribute isn’t the most reliable predictor of pass/fail.
The 4th attribute with the most information gain is what kind of job the mother has. The most common job was other which isn’t very specific.




Rank
Model
Accuracy
ROC Area

1
AdaBoostm1
72.91%
.667

2
Randomforest
70.13%
.677

3
Logistic Regression
68.61%
.671

4
Naive Bayes
68.86%
.683

5
j48
67.34%
.579
 
Given that we made a target variable (class) using the G3 grading scale we removed the G1, G2 and G3 variables from the dataset.  At this time we will run all of our models with the remaining variables in order to find which are the most informative overall.  Our hope is to be able to remove any variables with little to no effect on the predictive model.  
Initially, we ran four different algorithms using Weka in order to find which techniques could be the most helpful in predicting the target variable.  We ran a J48 tree (ranking previous failures, going out highest) Naivebayes, Logistic Regression (ranking failures, mjob-teacher highest), Randomforest, Adaboostm1 and compared the accuracy of each.  All of which were ran using default settings in Weka (cross validation - 10 fold).  We found all of them to be very similar in terms of accuracy with Adaboostm1 being the most accurate.   
Our first attempt to improve our model was to prune  the J48  tree.  The amount of variables we are working with could possibly be overfit in a J48 tree, to avoid this we produced a pruned model.  

The model had jumped from 67.34% correctly classified up to 69.62%, a gain of 2.28.  This improvement also put the j48 model in the top three overall.  
	
Our next attempt at improvement was to combine weekend and work week alcohol consumption.  We did this by multiplying weekend alcohol consumption by 2 and work week by 5, adding them together and dividing by 7.  This gave us an overall alcohol consumption during the week.  With this new variable we ran our top three previous models to compare :



Rank
Model
Correctly Classified %
ROC

1.
AdaBoostm1
71.39%
.676

2.
J48
70.37%
.667

3.
RandomForest
68.61%
.667

The models displayed above all performed less accurately than the original model with just the class variable and weekend/work week alcohol consumption as separate variables. 
	Our next attempt to improve the model was to remove the class variable of pass or fail.  Given that the grading system is based on a  0-20 scale we were hoping to analyze overall performance given that we had a small amount of failures.  The best way we thought to do this was using a simplelogistic algorithm in Weka.  We found that this model was very skewed as the ROC area was very high for only a couple of grades then much lower for the rest.  For this reason we do not feel comfortable that this model will help provide a reliable prediction.  

Evaluation - 
We stuck with our original Adaboost model. Our Adaboost model had solid results. It correctly classified 72.91% of the instances and had a ROC area of .676. It prevented overfitting by using the decision stump classifier. The decision stump is a weak learner and is less prone to overfitting compared to the J48 classifier and other classifiers.
A business case could be developed by a University in the United States. They could collect real data from their students and use the suggestions above to build a better model. They should observe more classrooms and enter the data into into a big set. With more instances they could create a better model and have a bigger training set when running the hold out in Weka. They should also add more attributes such as health and absences from class. They would have more possible predictors to use their our model. They could also enter the data in a better way. For example, failures was 1=1, 2=2, 3=3, 4+=4. If someone failed more than 4 times it was still entered in as a 4. By doing these things the university could build a great model that would be very useful to them. This model would help them get more ROI because students would stay in school longer and spend more money instead of dropping out.


Deployment - 
When a university/academy is looking to admit students it is obvious that they are searching for students that are most likely to fulfill their full time (4 years) at the university. They are looking to do this to increase total revenue over time, if a student has a short lifespan at the university their revenue for that student will be lower than that of the students that stay a full four years or further upon seeking greater education. Also if a student student fails an amount of courses it will be looked upon poorly for those potential new students and outside entities in the way that this institution has a higher fail/drop out rate. Lastly it costs an extreme amount for universities to market or campaign for new students to enter if some students have dropped out and no longer attend the university versus having a student stay for the full four years having no new recruitment costs. In addition to a reputation and prominence of the school, in some cases schools receive extra federal funding for high success rate and low drop out rate which can in turn will provide extra money to places that are needed. 

Nowadays when an institution is looking to admit new students into their school they are looking at such things like GPA, High School courses, Resumes, Work experience etc. We completely agree that these indicators do indeed give a great deal of insight as to how a student might perform in college, but in our analysis we have found that it can be very important and informative to look at other variables in one's life such as previous failures, parents education, etc. 

This is where our analysis comes into play. In our analysis we point out the common and highest influential variables that lead to class failures and in turn drop out rate. Some of these factors include previous failures, parents education and etc. We believe that by simply running a few models with a test set of the new applicant students a institution can greatly increase revenue, pass rate, and potential third party funding weather federal or private. 

Some risks associated with this proposition include: First of all this is just another way to look as potential applicants, and this is not a “fix all formula”. This algorithm or information should be used in align with all of the other criteria of the university as a new dimension of looking at candidates. If the university uses solely this criteria it will be very easy to pass over some students that are a good fit. Also some side notes is that the dataset we work with is kind of small. That being said some of the models we ran did fit well. Along with the dataset being small it is based out of European countries. With the dataset only containing instances of students it may not be applicable to other such institutions around the globe.  

Some potential ethical implications could include that if the institution too strongly relies on such factors as parents background/education it could potentially lead into bad ethics. I don't want to go as far as say it could lead to racism but with different upbringings it is potential that some could infer it that way. 

If anyone is curtios as to what the actual models look like I have screenshots for show if requested. Also have the Rstudtio code and Weka transformation code.

