library(stringdist)

phonetic("test")

names <- dbGetQuery(db, "
SELECT Name,
       Count
FROM NationalNames 
WHERE Gender = 'M'
AND Year = 2014")

head(names)

names$Soundex <- phonetic(names$Name)

head(names)

subset(names, Soundex == "J210")




