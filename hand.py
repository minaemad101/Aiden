import spotipy
import os
import dotenv
import numpy as np
import pickle
from spotipy.oauth2 import SpotifyOAuth
from rich import print
from methods import *
import asyncio
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import cv2
import mediapipe as mp
from skimage.feature import hog
import time

dotenv.load_dotenv()


scope = f"ugc-image-upload, user-read-playback-state, user-modify-playback-state, user-follow-modify, user-read-private, user-follow-read, user-library-modify, user-library-read, streaming, user-read-playback-position, app-remote-control, user-read-email, user-read-currently-playing, user-read-recently-played, playlist-modify-private, playlist-read-collaborative, playlist-read-private, user-top-read, playlist-modify-public"
print(os.getenv("SPOTIFY_CLIENT_ID"))
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=os.getenv("SPOTIFY_CLIENT_ID"), client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"), redirect_uri="http://localhost:8888/callback"), requests_timeout=300)

oldFreq=0
def getThr(img,checkHist):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Define lower and upper bounds for skin color in YCrCb
    if checkHist:
        cr_component = ycrcb[:, :, 1]
        # Flatten the Cr component array
        cr_flattened = cr_component.flatten()
        # Find the most frequent value
        most_frequent_value = np.argmax(np.bincount(cr_flattened))
        #print(most_frequent_value)
        v=max(125,most_frequent_value+5)
        v=min(v,140)
        lower_ycrcb = np.array([0,v , 85], dtype=np.uint8)   
        oldFreq=v
        #print(oldFreq)
    else:
        np.array([0, oldFreq, 85], dtype=np.uint8)   
        
    #lower_ycrcb = np.array([0, 140, 85], dtype=np.uint8)   
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    # Create a mask based on the YCrCb color range
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_OPEN, kernel)
    mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_CLOSE, kernel)
    extracted_skin = cv2.bitwise_and(img, img, mask=mask_ycrcb)
    # cv2.imshow('dst', extracted_skin)
    extracted_skin_gray = cv2.cvtColor(extracted_skin, cv2.COLOR_BGR2GRAY)   
    contours, hierarchy = cv2.findContours(extracted_skin_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box around largest contour
    if len(contours) > 0 :
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area<.05*400*400 :
            return [],[]
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Create a mask image for drawing contours
        mask = np.zeros_like(img)

        # Draw contours on the mask
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)
        crop_img = cv2.bitwise_and(img, mask)

        cropped_image = crop_img[y:y+h, x:x+w,:]
        b=cv2.resize(cropped_image,(400,400))
        return b,cnt
    return [],[]


        

v=False
def hand():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    i=0
    count=0

    temp=False
    temp_count=0
    s=-1
    go=True
    cap = cv2.VideoCapture(0)  # Open the webcam
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():
            if s!=-1 and time.time()-s<=2:
                go=False
                
            else:
                go=True
                s=-1
                
            success, image = cap.read()
            if not success:
                print("Failed to read video")
                break

            # Flip the image horizontally for a mirrored view
            #image = cv2.flip(image, 1)

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results = hands.process(image_rgb)
            res=-1
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks)==1 and go:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the landmarks on the image
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                    # Add the ring image on the middle finger (landmark index 12)
                    landmark1 = hand_landmarks.landmark[1]
                    landmark2 = hand_landmarks.landmark[2]
                    landmark3 = hand_landmarks.landmark[3]
                    landmark4 = hand_landmarks.landmark[4]
                    
                    landmark8 = hand_landmarks.landmark[8]
                    landmark7 = hand_landmarks.landmark[7]
                    landmark6 = hand_landmarks.landmark[6]
                    landmark5 = hand_landmarks.landmark[5]

                    landmark9 = hand_landmarks.landmark[9]
                    landmark10 = hand_landmarks.landmark[10]
                    landmark11 = hand_landmarks.landmark[11]
                    landmark12 = hand_landmarks.landmark[12]

                    landmark13 = hand_landmarks.landmark[13]
                    landmark14 = hand_landmarks.landmark[14]
                    landmark15 = hand_landmarks.landmark[15]
                    landmark16 = hand_landmarks.landmark[16]

                    landmark17 = hand_landmarks.landmark[17]
                    landmark18 = hand_landmarks.landmark[18]
                    landmark19 = hand_landmarks.landmark[19]
                    landmark20 = hand_landmarks.landmark[20]
                    f1=False
                    f2=False
                    f3=False
                    f4=False
                    f5=False
                    f0=False
                    if landmark8.y<landmark7.y and landmark7.y<landmark6.y and landmark6.y<landmark5.y:
                        #print("first -- ")
                        f1=True
                    if landmark12.y<landmark11.y and landmark11.y<landmark10.y and landmark10.y<landmark9.y:
                        #print("sec --")
                        f2=True
                    if landmark13.y>landmark14.y and landmark14.y>landmark15.y and landmark15.y>landmark16.y:
                        #print("thr --")
                        f3=True
                    if landmark17.y>landmark18.y and landmark18.y>landmark19.y and landmark19.y>landmark20.y:
                        #print("four --")
                        f4=True
                    print(abs(landmark9.x-landmark5.x))
                    if landmark1.y>landmark2.y and landmark2.y>landmark3.y and landmark3.y>landmark4.y and abs(landmark4.x-landmark5.x)>.11:
                        #print("five --")
                        f5=True
                    if landmark17.y>landmark13.y and landmark13.y>landmark9.y and landmark9.y>landmark5.y :
                        #print("five --")
                        f0=True
                    print(f1," ",f2," ",f3," ",f4," ",f5)
                    if f1 and f2 and f3 and f4 and f5:
                        print("five")
                        user_input=5
                    elif f1 and f2 and f3 and f4 and not f5:
                        print("four")
                        user_input=4
                    elif  f1 and f2 and f5 and not f3 and not f4:
                        print("three")
                        user_input=3
                    elif f1 and f2 and not f3 and not f4 and not f5:
                        print("two")
                        user_input=2
                    elif f1 and not f2 and not f3 and not f4 and not f5:
                        print("one")
                        user_input=1
                    elif f0 and not f1 and not f2 and not f3 and not f4 :
                        print("zero")
                        user_input=0
                    else:
                        print("nothing")
                        user_input=-1
                    if user_input == 0:
                    
                        if count==10 and i==0:
                            r=asyncio.run(pause_track(spotify=sp))
                            s=time.time()
                            count=0
                            temp=True
                            print(r)
                        if i==0 :
                            count+=1
                            i=0
                        else:
                            count=1
                            i=0    
                    elif user_input == 1:
                        if count==10 and i==1:
                            
                            r=asyncio.run(resume_track(spotify=sp))
                            
                            print(r)
                        
                            temp=True
                            s=time.time()
                        
                        if i==1 :
                            count+=1
                            i=1
                        else:
                            count=1
                            i=1



                    
                        
                    elif user_input == 2:
                        if count==10 and i==2:


                            current_volume = sp.current_playback()['device']['volume_percent']
                            new_volume = min(current_volume + 10, 100)


                            asyncio.run(change_volume(spotify=sp, volume=new_volume))
                            count=0
                            temp=True
                            print('Song is vol up playing.')
                            s=time.time()
                        
                        if i==2 :
                            count+=1
                            i=2
                        else:
                            count=1
                            i=2
                    elif user_input == 3:

                        if count==10 and i==3:
                            current_volume = sp.current_playback()['device']['volume_percent']
                            new_volume = max(current_volume - 10, 0)

                            asyncio.run(change_volume(spotify=sp, volume=new_volume))
                            count=0
                            temp=True
                            print('Song is vol down playing.')
                            s=time.time()
                        
                        if i==3 :
                            count+=1
                            i=3
                        else:
                            count=1
                            i=3
                    elif user_input == 4:
                        
                        if count==10 and i==4 :
                            
                            asyncio.run(play_previous_song(spotify=sp))
                            
                            count=0
                            s=time.time()
                            
                            temp=True

                                
                        if i==4 :
                            count+=1
                            i=4
                        else:
                            count=1
                            i=4
                    
                    elif user_input == 5:
                        if count==10 and i==5:
                            
                            asyncio.run(next_track(spotify=sp))
                            count=0
                            temp=True
                            s=time.time()
                        
                        if i==5 :
                            count+=1
                            i=5
                        else:
                            count=1
                            i=5
                    h, w, _ = image.shape
                
                        
                    landmark_points = []
                    
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        cv2.putText(image, str(idx), (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('th', image)
            k = cv2.waitKey(10)
            if k == 27:
                cv2.destroyAllWindows()
                break

