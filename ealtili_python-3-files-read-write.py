#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('writefile', 'test.txt', 'Hello, this is a quick test file.')




myfile = open('whoops.txt')




pwd




# Open the text.txt we made earlier
my_file = open('test.txt')




# We can now read the file
my_file.read()




# But what happens if we try to read it again?
my_file.read()




# Seek to the start of file (index 0)
my_file.seek(0)




# Now read again
my_file.read()




# Readlines returns a list of the lines in the file
my_file.seek(0)
my_file.readlines()




my_file.close()




# Add a second argument to the function, 'w' which stands for write.
# Passing 'w+' lets us read and write to the file

my_file = open('test.txt','w+')




# Write to the file
my_file.write('This is a new line')




# Read the file
my_file.seek(0)
my_file.read()




my_file.close()  # always do this when you're done with a file




my_file = open('test.txt','a+')
my_file.write('\nThis is text being appended to test.txt')
my_file.write('\nAnd another line here.')




my_file.seek(0)
print(my_file.read())




my_file.close()




get_ipython().run_cell_magic('writefile', '-a test.txt', '\nThis is text being appended to test.txt\nAnd another line here.')




get_ipython().run_cell_magic('writefile', 'test.txt', 'First Line\nSecond Line')




for line in open('test.txt'):
    print(line)




# Pertaining to the first point above
for asdf in open('test.txt'):
    print(asdf)

