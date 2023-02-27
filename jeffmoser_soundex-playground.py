library(stringdist)

phonetic("test")

head(names)

names$Soundex <- phonetic(names$Name)

head(names)

subset(names, Soundex == "J210")




