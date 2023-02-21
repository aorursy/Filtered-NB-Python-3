#!/usr/bin/env python
# coding: utf-8



# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(readr))
theme_set(theme_bw())

# read datasets
Questions <- suppressMessages(read_csv("../input/Questions.csv"))
Answers <- suppressMessages(read_csv("../input/Answers.csv"))
Tags <- suppressMessages(read_csv("../input/Tags.csv"))




library(tidytext)

title_words <- Questions %>%
    select(Id, Title, Score, CreationDate) %>%
    unnest_tokens(Word, Title)

head(title_words)




title_word_counts <- title_words %>%
    anti_join(stop_words, c(Word = "word")) %>%
    count(Word, sort = TRUE)

title_word_counts %>%
    head(20) %>%
    mutate(Word = reorder(Word, n)) %>%
    ggplot(aes(Word, n)) +
    geom_bar(stat = "identity") +
    ylab("Number of appearances in R question titles") +
    coord_flip()




Tags %>%
    count(Tag, sort = TRUE) %>%
    head(10)




common_tags <- Tags %>%
    group_by(Tag) %>%
    mutate(TagTotal = n()) %>%
    ungroup() %>%
    filter(TagTotal >= 100)

tag_word_tfidf <- common_tags %>%
    inner_join(title_words, by = "Id") %>%
    count(Tag, Word, TagTotal, sort = TRUE) %>%
    ungroup() %>%
    bind_tf_idf(Word, Tag, n)

# use only tags with at least 1000 questions

tag_word_tfidf %>%
    filter(TagTotal > 1000) %>%
    arrange(desc(tf_idf)) %>%
    head(10)




tag_word_tfidf %>%
    filter(Tag %in% c("ggplot2", "shiny", "data.table", "knitr")) %>%
    group_by(Tag) %>%
    top_n(10, tf_idf) %>%
    mutate(Word = reorder(Word, -tf_idf)) %>%
    ggplot(aes(Word, tf_idf)) +
    geom_bar(stat = "identity") +
    facet_wrap(~ Tag, scales = "free") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    ylab("TF-IDF")




title_words %>%
    anti_join(stop_words, c(Word = "word")) %>%
    group_by(Word) %>%
    summarize(Questions = n(),
              PercentPositive = mean(Score > 0)) %>%
    filter(Questions > 100) %>%
    ggplot(aes(Questions, PercentPositive)) +
    geom_point(size = .5) +
    geom_text(aes(label = Word), vjust = 1, hjust = 1, check_overlap = TRUE) +
    geom_hline(yintercept = mean(Questions$Score > 0), color = "red", lty = 2) +
    scale_x_log10(limits = c(50, 20000)) +
    scale_y_continuous(labels = scales::percent_format()) +
    xlab("# of occurences of word in title") +
    ylab("% of questions with this word that are positively scored")

