#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('lsmagic', '')
# You can add more on your local machine 




get_ipython().run_line_magic('pinfo', '%lsmagic')
# Use a questionmark to get a short description 




# Get random numbers to measure sorting time 
import numpy as np
n = 100000
random_numbers = np.random.random(size=n)

get_ipython().run_line_magic('time', 'random_numbers_sorted = np.sort(random_numbers)')
# Get execution time of a single line 




get_ipython().run_cell_magic('time', '', '# Get execution time of a whole cell (Has to be the first command in the cell) \n\nrandom_numbers = np.random.random(size=n)\nrandom_numbers_sorted = np.sort(random_numbers)')




get_ipython().run_line_magic('timeit', '-n 100 -r 5 random_numbers_sorted = np.sort(random_numbers)')
# n - execute the statement n times 
# r - repeat each loop r times and return the best 




get_ipython().run_cell_magic('timeit', '-n 100 -r 5', '# n - execute the statement n times \n# r - repeat each loop r times and return the best \n\nrandom_numbers = np.random.random(size=n)\nrandom_numbers_sorted = np.sort(random_numbers)')




get_ipython().run_cell_magic('prun', '', '# Returns a duration ranking of all called functions in the cell as well as a count for all funcition calls (Can only be seen by running it on your own) \n\nfor _ in range(5):\n    random_numbers = np.random.random(size=n)\n    random_numbers_sorted = np.sort(random_numbers)')




# Return working directory 
get_ipython().run_line_magic('pwd', '')




# Create a new folder 
get_ipython().run_line_magic('mkdir', "'test_folder'")




# Save to new .py file 
text = 'I am going into a new file'
get_ipython().run_line_magic('save', "'new_file' text")




# Copy files to a new location 
get_ipython().run_line_magic('cp', 'new_file.py new_file_2.py')




# List of elements in current directory 
get_ipython().run_line_magic('ls', '')




# Read7show files 
get_ipython().run_line_magic('cat', 'new_file.py')




# Remove folder 
get_ipython().run_line_magic('rmdir', 'test_folder/')

# Remove files 
get_ipython().run_line_magic('rm', 'new_file.py')




# Rename files 
get_ipython().run_line_magic('mv', 'new_file_2.py renamed_file.py')




get_ipython().run_line_magic('ls', '')




# Using a '!' neables arbitrary single-line bash-commands 
get_ipython().system('ls | grep .py')




get_ipython().run_cell_magic('!', '', '# This executes the whole cell in a bash and returns a list \npwd\nls')




get_ipython().run_cell_magic('bash', '', '# This executes the whole cell in a bash and returns single elements \npwd\nls')




# The returned values can be stored in variables 
working_directory = get_ipython().getoutput('pwd')
working_directory




# Compile latex in cells 




get_ipython().run_cell_magic('latex', '', '$\\frac{awe}{some}$')




# Html in cells 




get_ipython().run_cell_magic('HTML', '', '\n<h1>Awesome</h1>')




get_ipython().run_cell_magic('javascript', '', "\nwindow.alert('Here you can learn how to use magic functions inside notebooks.\\nHave a good day!')")






