#!/usr/bin/env python
# coding: utf-8



library(Rtsne)
library(data.table)
library(dplyr)
library(magrittr)

library(ggplot2) 
library(plotly)
library(ggthemes)

# system("ls ../input", intern = TRUE)
# Any results you write to the current directory are saved as output.




data <- fread("../input/creditcard.csv")




data %<>%
  mutate(id = 1:nrow(data)) %>%
  mutate(Class = as.integer(Class))

names(data) <- gsub('V', 'Feat', names(data))

numeric_interesting_features <- c(paste0('Feat', 1:28),
                                  'Amount') 
# "Class", the target, is not used to compute the 2D coordinates


data <- data[ apply(data, MARGIN = 1, FUN = function(x) !any(is.na(x))), ]




df <- (as.data.frame(data[numeric_interesting_features]))
# "Class", the target, is not used to compute the 2D coordinates

df_normalised <- apply(df, 
                       MARGIN = 2, 
                       FUN = function(x) {
                         scale(x, center = T, scale = T)
                       } )
df_normalised %<>%
  as.data.frame() %>%
  cbind(select(data, id))

# Remove line with potential NA
df_normalised <- df_normalised[ apply(df_normalised, MARGIN = 1, FUN = function(x) !any(is.na(x))), ]

data_fraud <- df_normalised %>%
    semi_join(filter(data, Class == 1), by = 'id')
  
data_sub <- df_normalised %>%
  sample_n(20000) %>% # sample of data
  rbind(data_fraud)
    
data_sub <- data_sub[!duplicated(select(data_sub, -id)), ]  # remove rows containing duplicate values within rounding




rtsne_out <- Rtsne(as.matrix(select(data_sub, -id)), pca = FALSE, verbose = TRUE,
                   theta = 0.3, max_iter = 1300, Y_init = NULL)
# "Class", the target, is not used to compute the 2D coordinates




# merge 2D coordinates with original features
tsne_coord <- as.data.frame(rtsne_out$Y) %>%
  cbind(select(data_sub, id)) %>%
  left_join(data, by = 'id') 




gg <- ggplot() +
  labs(title = "All Frauds (white dots) in the transaction landscape (10% of data)") +
  scale_fill_gradient(low = 'darkblue', high = 'red', name="Proportion\nof fraud per\nhexagon") +
  coord_fixed(ratio = 1) +
  theme_void() +
  stat_summary_hex(data = tsne_coord, aes(x=V1, y=V2, z = Class), bins=10, fun = mean, alpha = 0.9) +
  geom_point(data = filter(tsne_coord, Class == 0), aes(x = V1, y = V2), alpha = 0.3, size = 1, col = 'black') +
  geom_point(data = filter(tsne_coord, Class == 1), aes(x = V1, y = V2), alpha = 0.9, size = 0.3, col = 'white') +
  theme(plot.title = element_text(hjust = 0.5, family = 'Calibri'),
       legend.title.align=0.5)

  
gg
#On about 10% of the data

