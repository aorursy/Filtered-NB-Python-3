#!/usr/bin/env python
# coding: utf-8



# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql_advanced.ex4 import *
print("Setup Complete")




# Fill in your answer
query_to_optimize = 3

# Check your answer
q_1.check()




# Lines below will give you a hint or solution code
q_1.hint()
q_1.solution()




WITH CurrentOwnersCostumes AS
(
SELECT CostumeID 
FROM CostumeOwners 
WHERE OwnerID = MitzieOwnerID
),
OwnersCostumesLocations AS
(
SELECT cc.CostumeID, Timestamp, Location 
FROM CurrentOwnersCostumes cc INNER JOIN CostumeLocations cl
    ON cc.CostumeID = cl.CostumeID
),
LastSeen AS
(
SELECT CostumeID, MAX(Timestamp)
FROM OwnersCostumesLocations
GROUP BY CostumeID
)
SELECT ocl.CostumeID, Location 
FROM OwnersCostumesLocations ocl INNER JOIN LastSeen ls 
    ON ocl.timestamp = ls.timestamp AND ocl.CostumeID = ls.costumeID




# Lines below will give you a hint or the solution
q_2.hint()
q_2.solution()

