
# Face Recognition with OpenCV
# In[1]:


import cv2
import os
import numpy as np
import serial
import array
import time
from captcha.image import ImageCaptcha
from captcha.audio import AudioCaptcha
import matplotlib.pyplot as plt
import random
import speech_recognition as sr
import getpass
from easygui import passwordbox


password = passwordbox("What is your password ?")
if( password == "shushma"):
    print("welcome!!")
    arduinoData = serial.Serial('COM6',9600)
    temp1=1
    temp2=0
    j=0
    k=0
    t=0
    while(t!=2):
        print("\n")
        myData0 = (arduinoData.readline().strip())
        myData1 = (arduinoData.readline().strip())
        myData2 = (arduinoData.readline().strip())
        myData3 = (arduinoData.readline().strip())
        myData4 = (arduinoData.readline().strip())
        myData5 = (arduinoData.readline().strip())
        myData6 = (arduinoData.readline().strip())
        myData7 = (arduinoData.readline().strip())        
        a0=int(myData0)
        a1=int(myData1)
        a2=int(myData2)
        a3=int(myData3)
        a4=int(myData4)
        a5=int(myData5)
        a6=int(myData6)
        a7=int(myData7)
        arr = array.array('i', [a0, a1, a2, a3, a4, a5, a6, a7])
        temp2=0
        for i in range (8): 
            print(arr[i]),
            if (arr[i] > 35):
            #if(j == 0):
                if(j==1):
                    t=2
                temp2=1
        if (temp2 == 0):
            if(j==0):
                print ("System is Locked")
                j=1          
        
        if (t==2):
            print("\n")
            print("Found person..Syetem is getting ready for Reauntecation")
    # The number list, lower case character list and upper case character list are used to generate captcha text.
    number_list = ['0','1','2','3','4','5','6','7','8','9']

    alphabet_lowercase = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    alphabet_uppercase = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# In[2]:


    subjects = ["", "SHUSHMA","XYZ"]

# In[3]:

#function to detect face using OpenCV
    def detect_face(img):
    #   convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        #load OpenCV face detector, I am using LBP which is fast
        #there is also a more accurate but slow Haar classifier
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')       
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
        if (len(faces) == 0):
            return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
        (x, y, w, h) = faces[0]
    
    #return only the face part of the image
        return gray[y:y+w, x:x+h], faces[0]

# In[4]:

    def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
        faces = []
    #list to hold labels for all subjects
        labels = []
    
    
        for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
            if not dir_name.startswith("s"):
                continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
            label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
            for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

            #read image
                image = cv2.imread(image_path)
            
            #display an image window to show the image 
                #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                cv2.waitKey(100)
            
            #detect face
                face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
                if face is not None:
                    #add face to list of faces
                    faces.append(face)
                    #add label for this face
                    labels.append(label)
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    
        return faces, labels

# In[5]:


    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

#print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
# In[6]:

#create our LBPH face recognizer 
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# In[7]:

#train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))

# In[8]:

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# In[9]:

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the subject
    def predict(test_img):
        img = test_img.copy()
    #detect face from the image
        face, rect = detect_face(img)
        #print(face)    
        label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
        label_text = subjects[label]
    
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    
        return label_text

# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for. 

# In[10]:

    print("Predicting images...")

#own code for testing images from webcam
    cap = cv2.VideoCapture(0)
    num = 1 

    while num<2:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
        
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
            
                cv2.imwrite('test'+str(num)+'.jpg',img)
                num = num+1
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:

            break

    cap.release()
    cv2.destroyAllWindows()



#functions of speech recognition
# This function will create a random captcha string text based on above three list.
# The input parameter is the captcha text length.
    def create_random_captcha_text(captcha_string_size=8):
         captcha_string_list = []
         base_char = alphabet_uppercase 


   
         for i in range(captcha_string_size):

        # Select one character randomly.
            char = random.choice(base_char)
            print(char)
        # Append the character to the list.
            captcha_string_list.append(char)

         captcha_string = ''
    
    # Change the character list to string.    
         for item in captcha_string_list:
             captcha_string += str(item)

         return captcha_string

# This function will create a fully digital captcha string text.
    def create_random_digital_text(captcha_string_size=8):

        captcha_string_list = []
    # Loop in the number list and return a digital captcha string list
        for i in range(captcha_string_size):
            char = random.choice(number_list)
            captcha_string_list.append(char)
        
        captcha_string = ''

    # Convert the digital list to string.    
        for item in captcha_string_list:
            captcha_string += str(item)
        print(captcha_string)
        return captcha_string

# Create an image captcha with special text.
    def create_image_captcha(captcha_text):
        image_captcha = ImageCaptcha()
    # Create the captcha image.
        image = image_captcha.generate_image(captcha_text)
        # Save the image to a png file.
        image_file = "./captcha_"+captcha_text + ".png"
        image_captcha.write(captcha_text, image_file)

        r=sr.Recognizer()
    
    # Display the image in a matplotlib viewer.
        plt.show()
        plt.ion()
        plt.imshow(image)
        plt.draw()
        plt.pause(.001)
    #speech taking
        print("listening")
        flag=False
        with sr.Microphone() as source:
            audio=r.listen(source)
        plt.close()
        try:
            y=r.recognize_google(audio)
            print("System Predicts:"+ y )
        #print(type(y))
            z=str(y)
        #print(z)
            s=z.replace(" ", "")
            s=s.lower()
        #z.replace(" ","")
        #print(type(z))
            w=list(s)
            flag=True
        #print(w)
        except Exception:
            print("Something went Wrong")
    
    #print(image_file + " has been created.")
    #comparision of captcha and spoken
        m=captcha_text.lower()
        x=list(m)
    
        if (flag):
            f1=True
            if (x[7]!=w[0]):
                f1=False
            if (x[0]!=w[1]):
                f1=False
            if (x[5]!=w[2]):
                f1=False
            if (f1==True):
                print("matched!!")
                print("System Unlocked") 
            else:
                print("Not Matched!!")
                print("you are failed to say the password ,,please Login Again")


#load test images
    test_img1 = cv2.imread("test1.jpg")
#test_img2 = cv2.imread("test-data/test2.jpg")

#perform a prediction
    predicted_img1 = predict(test_img1)
#predicted_img2 = predict(test_img2)
    print("Prediction complete")
    print(predicted_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    if (predicted_img1=="SHUSHMA"):
    # Create random text.
        captcha_text = create_random_captcha_text()
    
    # Create image captcha.
        create_image_captcha(captcha_text)
    else:
        print("system doesnt recognised you ..Login again!!")
else:
    print("your password is incorrect.. pleaese try again!")