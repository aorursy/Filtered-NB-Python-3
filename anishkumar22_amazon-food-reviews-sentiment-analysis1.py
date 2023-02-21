#!/usr/bin/env python
# coding: utf-8



library(sentimentr)
library(stringr)
library(plyr)
library(dplyr)
library(ggplot2)
library(tm)
library(wordcloud)
library(RColorBrewer)




amr  <- read.csv("../input/Reviews.csv")




amr1 <- head(amr, 5000)
sentrv <- with(amr1, sentiment_by(Text))
qplot(amr1$Score, geom="histogram", binwidth = 1, main = "Histogram for Score", xlab = "Score", fill=I("blue"), col=I("red"), alpha=I(.8))




qplot(sentrv$ave_sentiment, geom="histogram", binwidth = 1, main = "Histogram for Avg Sentiment", xlab = "Avg Sentiment", fill=I("blue"), col=I("red"), alpha=I(.8))




summary(amr1$Score)




summary(sentrv$ave_sentiment)




best <- slice(amr1, top_n(sentrv, 5, ave_sentiment)$element_id)
bestt <- best[!duplicated(best$Text),]
bestt <- data.matrix(bestt$Text)
colnames(bestt) <- c("Top best reviews by sentiment score")
bestt




worst <- slice(amr1, top_n(sentrv, 5, -ave_sentiment)$element_id)
worstt <- worst[!duplicated(worst$Text),]
worstt <- data.matrix(worst$Text)
colnames(worstt) <- c("Top worst reviews by sentiment score")
worstt




amr2 <- head(amr, 5000)
stopw <- stopwords("english")
stopw <- stopw[stopw != "not"]

amrtext <- gsub('[[:punct:]]',' ', amr2$Text)
amrtext <- gsub('[[:cntrl:]]','', amrtext)
amrtext <- gsub('\\d+','', amrtext)
amrtext <- gsub('\n','', amrtext)
amrtext <- tolower( amrtext)

amrtext <- Corpus(VectorSource(amrtext))
amrtext1 <- tm_map(amrtext, removeWords, stopw)

amrtmtx <- DocumentTermMatrix(amrtext1)

amrtmtx1 <- removeSparseTerms(amrtmtx, 0.997)

aznwrdcld <- as.data.frame(as.matrix(amrtmtx1))

count <- colnames(aznwrdcld)
freq <- colSums(aznwrdcld)


wordcloud(count, freq, min.freq = sort(freq, decreasing = TRUE)[[600]],  colors=brewer.pal(8,"Dark2"), random.color=TRUE)


png("wordcloud.png")

