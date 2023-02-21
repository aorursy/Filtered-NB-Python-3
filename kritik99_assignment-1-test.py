#!/usr/bin/env python
# coding: utf-8



df_test_file="../input/test.txt.csv"
test<-read.csv("../input/test.txt.csv")
test

#STRUCTURE OF DATASET.....
str(test)  

#SUMMARY OF DATASET.....
summary(test)

#DIMENSION OF DATASET.....
dim(test)

#SETTING ANOTHER DATASET = OUR DATASET SO THAT OUR DATASET DON'T CHANGE..... 
test1<-test

#CALCULATING MEAN OF GIVEN FEATURES.....
mean(test1$LoanAmount[!is.na(test1$LoanAmount)])
mean(test1$Loan_Amount_Term[!is.na(test1$Loan_Amount_Term)])
mean(test1$ApplicantIncome[!is.na(test1$ApplicantIncome)])
mean(test1$CoapplicantIncome[!is.na(test1$CoapplicantIncome)])                                 
mean(test1$Credit_History[!is.na(test1$Credit_History)])

#CALCULATING MEDIAN OF GIVEN FEATURES.....
median(test1$LoanAmount[!is.na(test1$LoanAmount)])
median(test1$Loan_Amount_Term[!is.na(test1$Loan_Amount_Term)])
median(test1$ApplicantIncome[!is.na(test1$ApplicantIncome)])
median(test1$CoapplicantIncome[!is.na(test1$CoapplicantIncome)])                                 
median(test1$Credit_History[!is.na(test1$Credit_History)])

mode(test1$Loan_Amount_Term[!is.na(test1$Loan_Amount_Term)])
mode(test1$ApplicantIncome[!is.na(test1$ApplicantIncome)])
mode(test1$CoapplicantIncome[!is.na(test1$CoapplicantIncome)])                                 
mode(test1$Credit_History[!is.na(test1$Credit_History)])

#IMPUTING MISSING VALUES WITH MEAN.....
test1$LoanAmount[is.na(test1$LoanAmount)]<-mean(test1$LoanAmount[!is.na(test1$LoanAmount)])
test1$LoanAmount
summary(test1)

test1$Loan_Amount_Term[is.na(test1$Loan_Amount_Term)]<-mean(test1$Loan_Amount_Term[!is.na(test1$Loan_Amount_Term)])
test1$Loan_Amount_Term

#IMPUTING MISSING VALUES WITH MEDIAN.....
test1$Credit_History[is.na(test1$Credit_History)]<-median(test1$Credit_History[!is.na(test1$Credit_History)])
test1$Credit_History

#USING KNN METHOD ON loan amount

install.packages("VIM")
library(VIM)
test3<- kNN(test,variable=c("LoanAmount"),k=5)
summary(test3)

#HISTOGRAM AND DENSITY PLOT....
hist(test1$ApplicantIncome)
hist(test1$ApplicantIncome,freq = FALSE)

hist(test1$CoapplicantIncome)
hist(test1$CoapplicantIncome,freq = FALSE)

hist(test1$LoanAmount)
hist(test1$LoanAmount,freq = FALSE)

hist(test1$Loan_Amount_Term)
hist(test1$Loan_Amount_Term,freq = FALSE)


install.packages("ggplot2")

library(ggplot2)

#PLOT.....
ggplot(test1,aes(x=Dependents,y=ApplicantIncome,col=Gender))+
  geom_point()

ggplot(test1,aes(x=Loan_Amount_Term,y=LoanAmount))+
  geom_point()

# BAR AND BOXPLOT
ggplot(test1,aes(x=Dependents,y=ApplicantIncome,col=Gender))+
  geom_bar(stat="summary",fun.y="mean")

ggplot(test1,aes(x=Dependents,y=ApplicantIncome,fill=Gender))+
  geom_bar(stat="summary",fun.y="mean",col="black")

boxplot(test1)
length(test1)

ggplot(test1,aes(x=Dependents,y=ApplicantIncome,fill=Gender))+
  geom_bar(stat="summary",fun.y="mean",col="black")+
  geom_boxplot(fill="red",col="pink",notch=TRUE)
  
#enhancing our plot(we can extract many things from this plot if we 
#have to compare on the basis of gender

mytest<-ggplot(test1,aes(x=Dependents,y=ApplicantIncome,fill=Gender))+
  geom_bar(stat="summary",fun.y="mean",col="black")+
  geom_point(position = position_jitter(0.3),size=2,shape=21)

mytest


#CREATING NEW FEATURES(APPLICANT INCOME IN THIS CASE)
test2<-test1
mark1<-quantile(test2$ApplicantIncome,0.25)
mark2<-quantile(test2$ApplicantIncome,0.5)
mark3<-quantile(test2$ApplicantIncome,0.75)
mark4<-quantile(test2$ApplicantIncome,1)

#ASSIGNING LOWER CLASS, LOWER MIDDLE CLASS, UPPER MIDDLE CLASS & UPPER CLASS
#in this i was not able to understand why some entries (example entry 19) doesn't change
test2$ApplicantIncome[test2$ApplicantIncome<mark1]<-"Lower Class"
test2$ApplicantIncome[test2$ApplicantIncome<mark2]<-"Lower Middle class"
test2$ApplicantIncome[test2$ApplicantIncome<mark3]<-"Upper Middle Class"
test2$ApplicantIncome[test2$ApplicantIncome<mark4]<-"Upper Class"

test2$ApplicantIncome


#DEALING WITH OUTLIERS
boxplot(test1$ApplicantIncome)#it has outliers
boxplot(test1$CoapplicantIncome)#it has outliers
boxplot(test1$LoanAmount)#it has outliers

#i am using outlier treatment BY WINSORIZING METHOD
#FOR ApplicantIncome
Q3<-quantile(test1$ApplicantIncome,0.75)
check1<-Q3+ 1.5*IQR(test1$ApplicantIncome)

test1$ApplicantIncome[test1$ApplicantIncome>check1]<-check1
boxplot(test1$ApplicantIncome)

#FOR CoapplicantIncome
Q33<-quantile(test1$CoapplicantIncome,0.75)
check1<-Q33+ 1.5*IQR(test1$CoapplicantIncome)

test1$CoapplicantIncome[test1$CoapplicantIncome>check1]<-check1
boxplot(test1$CoapplicantIncome)

#FOR LoanAmount
Q33<-quantile(test1$LoanAmount,0.75)
check1<-Q33+ 1.5*IQR(test1$LoanAmount)

test1$LoanAmount[test1$LoanAmount>check1]<-check1
boxplot(test1$LoanAmount)





















