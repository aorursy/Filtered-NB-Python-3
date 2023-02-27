#!/usr/bin/env python
# coding: utf-8



# d <- read.csv('../input/movie_metadata.csv')
str(d)
attach(d)

## No of movies relesed per year

plot(table(title_year))




#As we can see, with time the number of movie released has been increased. Now, we move further, we will look on the the money spend on movie (average) each year. 




## Money spend on  each movies (average per year)

c=rep(0,101)
i=1
for(e in 1916:2016){
    h<-sum(title_year==e,na.rm=TRUE)
    j<-sum(budget[title_year==e],na.rm=TRUE)
    c[i]<-j/h
    i=i+1
}
e=1916:2016
plot(e,c,xlab="Year",ylab="Average money on movie")




#The output from this plot is that the budget has been increased on movie with time. Now its time to look into the average IMDB score of each year




## Average of IMDB score with year

c=rep(0,101)
i=1
for(e in 1916:2016){
    c[i]<- mean(imdb_score[title_year==e],na.rm=TRUE)
    i=i+1
}
e=1916:2016
plot(e,c,xlab="Year",ylab="Average IMDB")




#the conclusion from the plot is the decrease mean value ofIMDB score with time. So the increase in number of movies made each year, has deteriorated the quality. Next plot is about the number of users voted for each movie vs its IMDB score. 




## No of users per movie

plot(num_voted_users,imdb_score)




#This is the lastr plot, where i tried to find out the number of commercial sucess movies over the year by subtracting the gross with budget. 




# CS<- which(gross-budget >0)
#the number of hit movies
length(CS)
# FL<- which(gross-budget <0)
#the number of flop movies
length(FL)
#mean of hit movies IMDB score
mean(imdb_score[CS])
#mean of flop movies IMDB score
mean(imdb_score[FL])




#So the number of commercial sucees movies is more than that of flops, but their mean IMDB score turns out to be equal(almost). Thud it can be concluded that the box office sucess does not reflect on the IMDB score of a movie. They bothn atre different.




#Now i will plot the graph of profit/loss of a movie vs its IMDB score to look for any pattern in between them
    




plot(gross-budget,imdb_score,xlab="profit/loss")




#This plot is not giving the desired result (except that a korean movie named "The Host" made the highest loss ever (left most part)). So now i will plot the 1st quartile to 3rd quartile part of "gross-budget" to lower the range.




s<- summary(gross-budget)
g<-((s[2]<gross-budget)&(s[5]>gross-budget))
plot((gross-budget)[g],imdb_score[g],xlab="Profit/loss",ylab="IMDB score")




#the plot is scattered all over without any pattern, Thus it can be concluded that the flop and commercial suceess both can have a good or bad IMDB score.

