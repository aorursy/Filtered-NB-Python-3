#!/usr/bin/env python
# coding: utf-8



homi.r <- read.csv("../input/database.csv")
suppressMessages(attach(homi.r))
options(warn=-1)
head(homi.r)




# Libraries

# warn.conflicts = FALSE
suppressMessages(library(plyr))
suppressMessages(library(tidyr))
suppressMessages(library(stringr))
suppressMessages(library(stringi))
suppressMessages(library(dplyr))
suppressMessages(library(forcats))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))
suppressMessages(library(Hmisc))




# str(homi.r)
# dim(homi.r)
# [1] 638454     24
# class(homi.r)
# [1] "data.frame"
# length(unique(homi.r$Record.ID))
# [1] 638454
# anyDuplicated(homi.r$Record.ID)
# [1] 0
# anyNA(homi.r)
# [1] TRUE
# names(homi.r)

# [1] "Record.ID"             "Agency.Code"           "Agency.Name"           "Agency.Type"          
# [5] "City"                  "State"                 "Year"                  "Month"                
# [9] "Incident"              "Crime.Type"            "Crime.Solved"          "Victim.Sex"           
# [13] "Victim.Age"            "Victim.Race"           "Victim.Ethnicity"      "Perpetrator.Sex"      
# [17] "Perpetrator.Age"       "Perpetrator.Race"      "Perpetrator.Ethnicity" "Relationship"         
# [21] "Weapon"                "Victim.Count"          "Perpetrator.Count"     "Record.Source"

##########################




names(homi.r) <- tolower(names(homi.r))
# names(homi.r)
# [1] "record.id"             "agency.code"           "agency.name"           "agency.type"          
# [5] "city"                  "state"                 "year"                  "month"                
# [9] "incident"              "crime.type"            "crime.solved"          "victim.sex"           
# [13] "victim.age"            "victim.race"           "victim.ethnicity"      "perpetrator.sex"      
# [17] "perpetrator.age"       "perpetrator.race"      "perpetrator.ethnicity" "relationship"         
# [21] "weapon"                "victim.count"          "perpetrator.count"     "record.source"   




# sapply(homi.r[1,],class)

# record.id           agency.code           agency.name           agency.type                  city 
# "integer"              "factor"              "factor"              "factor"              "factor" 
# state                  year                 month              incident            crime.type 
# "factor"             "integer"              "factor"             "integer"              "factor" 
# crime.solved            victim.sex            victim.age           victim.race      victim.ethnicity 
# "factor"              "factor"             "integer"              "factor"              "factor" 
# perpetrator.sex       perpetrator.age      perpetrator.race perpetrator.ethnicity          relationship 
# "factor"             "integer"              "factor"              "factor"              "factor" 
# weapon          victim.count     perpetrator.count         record.source 




# Ok! I think I will create new data frame filtering the crime solved to "YES", 
# i want to go from what we know to what we don’t

# levels(homi.r$crime.solved)
# [1] "No"  "Yes"
# no missing data in this feature, Great!

# length(homi.r$crime.solved[homi.r$crime.solved == "Yes"])
# [1] 448172 
# 448172 / 638454 = 0.70  realy nice 70 %




###### crime solved = yes

homi.r.solved <- homi.r %>% filter(
                                    crime.solved    == "Yes" &
                                    victim.sex      != "Unknown" &
                                    perpetrator.sex != "Unknown" &
                                    relationship    != "Unknown"
)%>%
  droplevels()

##############################################
girl.boy.crime<-homi.r %>% filter(
                                    relationship    == c("Girlfriend","Boyfriend") &
                                    victim.age      >= 18 &
                                    perpetrator.age >= 18 &
                                    victim.sex      != "Unknown" &
                                    perpetrator.sex != "Unknown"
) %>%
  droplevels() %>%
  select(record.id, agency.type,city,state,year,month,crime.type,
         crime.solved,victim.sex,victim.age,victim.race,victim.ethnicity,perpetrator.sex,
         perpetrator.age,perpetrator.race,perpetrator.ethnicity,relationship,weapon,record.source)

# table(girl.boy.crime$victim.sex)
# Female   Male 
#   7468   3619
# table(girl.boy.crime$perpetrator.sex)
# Female   Male 
#   3460   7627

# something wrong !!!

girl.boy.crime<-girl.boy.crime[!(girl.boy.crime$victim.sex==girl.boy.crime$perpetrator.sex),]

# table(girl.boy.crime$victim.sex)
# Female   Male 
# 7434   3426 
# table(girl.boy.crime$perpetrator.sex)
# Female   Male 
# 3426   7434 

# now it is OK 

############################################

# Who killed Who (Familicide). Unfortunately !


homi.r.solved$who.killed.who <- ifelse(homi.r.solved$relationship=="Brother" &
                                         homi.r.solved$perpetrator.sex =="Male" ,
                                       "Brother Killed by Brother",
                                       ifelse(homi.r.solved$relationship=="Brother" &
                                                homi.r.solved$perpetrator.sex =="Female" ,
                                              "Brother Killed by Sister",
                                              ifelse(homi.r.solved$relationship=="Sister" &
                                                       homi.r.solved$perpetrator.sex =="Female" ,
                                                     "Sister Killed by Sister",
                                                     ifelse(homi.r.solved$relationship=="Sister" &
                                                              homi.r.solved$perpetrator.sex =="Male" ,
                                                            "Sister Killed by Brother",
                                                            ifelse(homi.r.solved$relationship=="Father" &
                                                                     homi.r.solved$perpetrator.sex =="Female" ,
                                                                   "Father Killed by Daughter",
                                                                   ifelse(homi.r.solved$relationship=="Father" &
                                                                            homi.r.solved$perpetrator.sex =="Male" ,
                                                                          "Father Killed by Sun",
                                                                          ifelse(homi.r.solved$relationship=="Mother" &
                                                                                   homi.r.solved$perpetrator.sex =="Male" ,
                                                                                 "Mother Killed by Sun",
                                                                                 ifelse(homi.r.solved$relationship=="Mother" &
                                                                                          homi.r.solved$perpetrator.sex =="Female" ,
                                                                                        "Mother Killed by Daughter",
                                                                                        ifelse(homi.r.solved$relationship=="Wife" &
                                                                                                 homi.r.solved$perpetrator.sex =="Male" ,
                                                                                               "Wife Killed by Husband",
                                                                                               ifelse(homi.r.solved$relationship=="Husband" &
                                                                                                        homi.r.solved$perpetrator.sex =="Female" ,
                                                                                                      "Husband Killed by Wife",
                                                                                                      ifelse(homi.r.solved$relationship=="Son" &
                                                                                                               homi.r.solved$perpetrator.sex =="Female" ,
                                                                                                             "Son Killed by Mother",
                                                                                                             ifelse(homi.r.solved$relationship=="Son" &
                                                                                                                      homi.r.solved$perpetrator.sex =="Male" ,
                                                                                                                    "Son Killed by Father","UKN"
                                              ))))))))))))
# Im sure there is a simpler way to do the same, but really i do not have Time ..!

ex.husband.ex.wife.crime<-homi.r %>% filter(
                                            relationship     == c("Ex-Husband","Ex-Wife") &
                                            victim.age       > 18 &
                                            perpetrator.age  > 18 &
                                            victim.sex      != "Unknown" &
                                            perpetrator.sex != "Unknown"
) %>%
  droplevels() %>%
  select(record.id, agency.type,city,state,year,month,crime.type,
         crime.solved,victim.sex,victim.age,perpetrator.sex,
         perpetrator.age,relationship,weapon)

# there are Female in both features victim .sex and perpetrator.sex ..and Male as well which is wrong..!
# so I consider that the victim sex to be my reference because it is the subject of this dataset.
ex.husband.ex.wife.crime$relationship <- ifelse(ex.husband.ex.wife.crime$victim.sex=="Male",
                                                "Ex-Husband","Ex-Wife")
# I think there was some typo here 

# OK let's move on

ex.husband.ex.wife.crime$older.or.younger <- ifelse(ex.husband.ex.wife.crime$relationship=="Ex-Husband" &
                                                      ex.husband.ex.wife.crime$perpetrator.age < ex.husband.ex.wife.crime$victim.age,
                                                    "Ex-wife Killed an old Ex-Husband",
                                                    ifelse(ex.husband.ex.wife.crime$relationship=="Ex-Husband" &
                                                             ex.husband.ex.wife.crime$perpetrator.age > ex.husband.ex.wife.crime$victim.age,
                                                           "Ex-wife Killed a young Ex-Husband",
                                                           ifelse(ex.husband.ex.wife.crime$relationship=="Ex-Wife" &
                                                                    ex.husband.ex.wife.crime$perpetrator.age > ex.husband.ex.wife.crime$victim.age,
                                                                  "Ex- Husband Killed a young Ex-Wife",
                                                                  ifelse(ex.husband.ex.wife.crime$relationship=="Ex-Wife" &
                                                                           ex.husband.ex.wife.crime$perpetrator.age < ex.husband.ex.wife.crime$victim.age,
                                                                         "Ex-Husband Killed an old Ex-Wife","Smae Age"))))




homi.r$who.killed.who.sex <- ifelse(homi.r$perpetrator.sex=="Female"& homi.r$victim.sex=="Male",
                                    "Male Killed by Female",
                                    ifelse(homi.r$perpetrator.sex=="Male"& homi.r$victim.sex=="Female",
                                          "Female Killed by Male",
                                          ifelse(homi.r$perpetrator.sex =="Male" & homi.r$victim.sex == "Male",
                                                 "Male Killed by Male",
                                                 ifelse(homi.r$perpetrator.sex =="Female" & homi.r$victim.sex == "Female",
                                                        "Female Killed by Female", "UNK"))))




employee.employer.crime<-homi.r %>% filter(
                                           relationship     == c("Employee","Employer") &
                                           victim.age       > 18 &
                                           perpetrator.age  > 18 &
                                           victim.sex      != "Unknown" &
                                           perpetrator.sex != "Unknown"
) %>%
  droplevels() %>%
  select(record.id, agency.type,city,state,year,month,crime.type,
         crime.solved,victim.sex,victim.age,perpetrator.sex,
         perpetrator.age,relationship,weapon,who.killed.who.sex)




# table(homi.r.solved$who.killed.who)
# Brother Killed by Brother  Brother Killed by Sister Father Killed by Daughter      Father Killed by Sun 
# 5016                       480                       473                      3880 
# Husband Killed by Wife Mother Killed by Daughter      Mother Killed by Sun  Sister Killed by Brother 
# 8613                       686                      3549                      1037 
# Sister Killed by Sister      Son Killed by Father      Son Killed by Mother                       UKN 
# 251                      5983                      3859                    296279 
# Wife Killed by Husband 
# 23055 




by.year <- summarise(group_by(homi.r,year),freq.year =n())%>%
  arrange(desc(freq.year)) 

by.month <- summarise(group_by(homi.r,month),freq.month =n())%>%
  arrange(desc(freq.month))


by.family <- summarise(group_by(homi.r.solved[homi.r.solved$who.killed.who!="UKN", ],who.killed.who),total.number.re =n())%>%
    arrange(desc(total.number.re))

by.state <- summarise(group_by(homi.r,state),freq.by.state =n())%>%
  arrange(desc(freq.by.state))




empyr.empee.sex <- employee.employer.crime %>%
                                                group_by(victim.sex, 
                                                perpetrator.sex,
                                                who.killed.who.sex,
                                                 relationship) %>%
                  summarise(sex.freq = n()) %>%
                  arrange(victim.sex, perpetrator.sex)




#################################################
homi.theme<-theme(
  axis.text = element_text(size = 8),
  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 0.5),
  axis.title = element_text(size = 14),
  panel.grid.major = element_line(color = "grey"),
  panel.grid.minor = element_blank(),
  panel.background = element_rect(fill = "snow1"),
  legend.position = "right",
  legend.justification = "top", 
  legend.background = element_blank(),
  panel.border = element_rect(color = "black", fill = NA, size = 1))
####################################################

 tt.homi.f <- ttheme_minimal(
    core=list(bg_params = list(fill = "azure", col="darkblue"),
              fg_params=list(fontface=6)),
    colhead=list(fg_params=list(col="navyblue", fontface=4L)))




ggplot(empyr.empee.sex ,aes(x = who.killed.who.sex, y=sex.freq, fill=relationship))+
  geom_bar(stat="identity", alpha=0.4,col="gold", width=0.4)+facet_wrap(~ relationship)+
  homi.theme+
  ggtitle("Victims \n Employee VS Employer \n Male VS Female \n Who Killed by Who?")+
  labs(x= "Who Killed by Who",
       y= "Number of incidents")




ggplot(employee.employer.crime ,aes(x=relationship,fill = relationship))+
  geom_bar(alpha=0.4,col="gold", width=0.4)+
  homi.theme+
  ggtitle("Victims \n Employee VS Employer")+labs(x= "Relationship",
                                                  y= "Number of incidents")




ex.h.vs.ex.w <- ggplot(ex.husband.ex.wife.crime ,aes(x=relationship,fill = relationship))+
  geom_bar(alpha=0.4,col="gold", width=0.4)+
  homi.theme+
  ggtitle("Victims \n Ex-Husband VS Ex-Wife")+labs(x= "Relationship",
                                                   y= "Number of incidents")




ex.h.vs.ex.w.age <- ggplot(ex.husband.ex.wife.crime, aes(x= older.or.younger))+geom_bar(alpha=0.7,fill="gold3", width=0.3)+
  homi.theme+
  ggtitle("Victims \n Ex-Husband VS Ex-Wife VS Age")+
  labs(x="Older and Yonger",
       y="Number of Incidents")+
  theme(axis.text.x=element_text(size= 8, angle=90,hjust = 0.5))





grid.arrange(ex.h.vs.ex.w,
             ex.h.vs.ex.w.age,ncol=2)




by.family$who.killed.who<- fct_inorder(by.family$who.killed.who)  
  
  plot.by.family <- ggplot(by.family,aes(x=who.killed.who, y=total.number.re ))+geom_bar(stat="identity", fill="darkred",width = 0.5)+
    theme(axis.text.x=element_text(size= 8, angle=90,hjust = 0.5))+
    homi.theme+
    ggtitle("Who Killed Who! \n \n Number of Incidents VS Family Relationship")+
    labs(x="Family relationship",
         y="Number of Incidents")
 
 table.by.family <- tableGrob(by.family, rows=NULL,theme = tt.homi.f)
 
 grid.arrange(plot.by.family,
              table.by.family,ncol=2)




plot.gf.bf.vic<-ggplot(girl.boy.crime ,aes(x=victim.sex,fill=victim.sex))+
  geom_bar(alpha=0.4,col="gold", width=0.4)+
  homi.theme+
  ggtitle("Victims \n Girlfriend VS Boyfriend")+labs(x= "Gender",
                              y= "Number of incidents")

plot.gf.bf.pre<-ggplot(girl.boy.crime ,aes(x=perpetrator.sex, fill=perpetrator.sex))+
  geom_bar(alpha=0.4,col="gold", width = 0.4)+
  homi.theme+
  ggtitle("Perpetrators \n Girlfriend VS Boyfriend")+labs(x= "Gender",
                                     y= "Number of incidents")

grid.arrange(plot.gf.bf.vic,
             plot.gf.bf.pre,
             ncol=2)




plot.homic.years<-ggplot(data = by.year,
                       aes(x=as.numeric(year),
                           y=freq.year))+
  geom_line(size=2,col="yellow3")+
  homi.theme+
  ggtitle("Number of incidents occurred per Year")+
  labs(x="Year",
       y="Number of Incidents")+
  theme(axis.text.x=element_text(size= 5, angle=90,hjust = 0.5))

plot.homic.years.points<-ggplot(data = by.year,
                              aes(x=as.factor(year),
                                  y=freq.year))+
  geom_point(size=1,col="blue")+
  homi.theme+
  ggtitle("Number of Incidents occurred per Year")+
  labs(x="Year",
       y="Number of Incidents")+
  theme(axis.text.x=element_text(size= 5, angle=90,hjust = 0.5))


by.state$state <- fct_inorder(by.state$state)
plot.by.state <- ggplot(data = by.state,
                        aes(x=as.factor(state),
                            y=freq.by.state))+
  geom_bar(stat= "identity", fill="darkred", width=0.5 )+
  homi.theme+
  ggtitle("Number of Incidents occurred per Year")+
  labs(x="State",
       y="Number of Incidents")+
  theme(axis.text.x=element_text(size= 6, angle=90,hjust = 0.5))

grid.arrange(arrangeGrob(plot.homic.years,plot.homic.years.points,ncol=2),
             plot.by.state)

