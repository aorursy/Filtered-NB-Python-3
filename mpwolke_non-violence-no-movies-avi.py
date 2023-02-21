#!/usr/bin/env python
# coding: utf-8



pip install MoviePy




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import moviepy.editor as moviepy
clip = moviepy.VideoFileClip("/kaggle/input/moviesviolencenonviolence/movies/NonViolence/football_crowds__Giants_of_Brazil_5_of_6__anandaliyanage__5tw2pojykz8.avi")
clip.write_videofile("/kaggle/input/moviesviolencenonviolence/movies/NonViolence/football_crowds__Giants_of_Brazil_5_of_6__anandaliyanage__5tw2pojykz8.mp4")




get_ipython().run_line_magic('matplotlib', 'inline')




from IPython.display import Video

Video("/kaggle/input/moviesviolencenonviolence/movies/NonViolence/football_crowds__Giants_of_Brazil_5_of_6__anandaliyanage__5tw2pojykz8.avi")




import numpy as np
import cv2
import matplotlib.pyplot as plt

# read video
cap = cv2.VideoCapture('/kaggle/input/moviesviolencenonviolence/movies/NonViolence/football_crowds__Giants_of_Brazil_5_of_6__anandaliyanage__5tw2pojykz8.avi')

ret, frame = cap.read()    
plt.figure()
plt.imshow(frame)




import numpy as np
import cv2
import matplotlib.pyplot as plt

# read video
cap = cv2.VideoCapture('/kaggle/input/moviesviolencenonviolence/movies/NonViolence/stadium_crowds__Maracana_Football_Crowd_Atmosphere__thetravelmap__xfi3rvFKIPU.avi')

ret, frame = cap.read()    
plt.figure()
plt.imshow(frame)




import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('kaggle/input/moviesviolencenonviolence/movies/NonViolence/football_crowds__Giants_of_Brazil_5_of_6__anandaliyanage__5tw2pojykz8.avi')


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()




#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;
 
int main(){
 
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("kaggle/input/moviesviolencenonviolence/movies/NonViolence/football_crowds__Giants_of_Brazil_5_of_.avi"); 
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
     
  while(1){
 
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
 
    // Display the resulting frame
    imshow( "Frame", frame );
 
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
  
  (/, When, everything, done,, release, the, video, capture, object)
  cap.release();
 
  (/, Closes, all, the, frames)
  destroyAllWindows();
     
  return 0;
}




#Code by Olga Belitskaya https://www.kaggle.com/olgabelitskaya/sequential-data/comments
from IPython.display import display,HTML
c1,c2,f1,f2,fs1,fs2='#a83a32','#a8324e','Akronim','Smokum',30,15
def dhtml(string,fontcolor=c1,font=f1,fontsize=fs1):
    display(HTML("""<style>
    @import 'https://fonts.googleapis.com/css?family="""\
    +font+"""&effect=3d-float';</style>
    <h1 class='font-effect-3d-float' style='font-family:"""+\
    font+"""; color:"""+fontcolor+"""; font-size:"""+\
    str(fontsize)+"""px;'>%s</h1>"""%string))
    
    
dhtml('Kaggle Notebook Runner: Marília Prata, not a DS. Shh! @mpwolke' )

