import os

os.listdir(".")

with open("config.json") as c:
    data = c.read()
data


