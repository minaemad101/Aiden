import spotipy
import os
import dotenv
from spotipy.oauth2 import SpotifyOAuth
from rich import print
from methods import *
import asyncio
from src.initialize_speech import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from src.nlp import *
import cv2
from skimage.feature import hog
import time


dotenv.load_dotenv()

scope = f"ugc-image-upload, user-read-playback-state, user-modify-playback-state, user-follow-modify, user-read-private, user-follow-read, user-library-modify, user-library-read, streaming, user-read-playback-position, app-remote-control, user-read-email, user-read-currently-playing, user-read-recently-played, playlist-modify-private, playlist-read-collaborative, playlist-read-private, user-top-read, playlist-modify-public"
print(os.getenv("SPOTIFY_CLIENT_ID"))
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=os.getenv("SPOTIFY_CLIENT_ID"), client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"), redirect_uri="http://localhost:8888/callback"), requests_timeout=300)



def getThr(img,checkHist,oldFreq):
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
       lower_ycrcb = np.array([0, oldFreq, 85], dtype=np.uint8)   
        
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
            return [],[],oldFreq
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Create a mask image for drawing contours
        mask = np.zeros_like(img)

        # Draw contours on the mask
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)
        crop_img = cv2.bitwise_and(img, mask)

        cropped_image = crop_img[y:y+h, x:x+w,:]
        b=cv2.resize(cropped_image,(400,400))
        return b,cnt,oldFreq
    return [],[],oldFreq



def hand_2():
    with open('./src/models/modellogisiticnewwith093.pickle', 'rb') as file:   #modellogisiticwith087
        modelL = pickle.load(file)
    with open('./src/models/modelRandomnewwith093.pickle', 'rb') as file:      #modelRandomwith091
        modelR = pickle.load(file)
    with open('./src/models/modellogisiticface.pickle', 'rb') as file:   #modellogisiticwith087
        modelLFace = pickle.load(file)
    with open('./src/models/modelRandomface.pickle', 'rb') as file:   #modellogisiticwith087
        modelRFace = pickle.load(file)
    cap = cv2.VideoCapture(0)  #"t9.mp4"
    i=0
    count=0

    temp=False
    temp_count=0
    s=-1
    checkHist=True
    countCheck=0
    oldFreq=0
    while True:
        ret, frame = cap.read()
        
        if ret:
            frame=cv2.flip(frame, 1)
            #cv2.imwrite("output/n"+str(countCheck)+".jpg",frame)
            thresh1,cnt,oldFreq=getThr(frame,checkHist,oldFreq)

            if countCheck%100==0:
                checkHist=True
            else:
                checkHist=False
            

        

            
            if len(thresh1)==0:
                continue
            cv2.imwrite("output/thr"+str(countCheck)+".jpg",thresh1)
            thresh1=cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
            
            countCheck+=1
            hog_features = hog(thresh1, orientations=12, pixels_per_cell=(12, 12), cells_per_block=(4, 4))
    
            y_predLFace=modelLFace.predict_proba([hog_features])
            y_predRFace=modelRFace.predict_proba([hog_features])
        
            
            if y_predLFace[0][1]>.9 and y_predRFace[0][1]>.5 :
                #print("log=",y_predLFace[0],"   rand=",y_predRFace[0])
                y_predL=modelL.predict_proba([hog_features])
                y_predR=modelR.predict_proba([hog_features])
                user_inputL=np.argmax(y_predL)
                user_inputR=np.argmax(y_predR)
                if s!=-1 and time.time()-s<=2:
                    
                    y_predL[0][user_inputL]=0
                else:
                    s=-1
                # print(user_inputL,"  ",y_predL[0][user_inputL],"     R= ",user_inputR,"  ",y_predR[0][user_inputR])
                if y_predL[0][user_inputL]>.85 and y_predR[0][user_inputR]>.4 and user_inputL==user_inputR:
                    # print(user_inputL,"  ",y_predL[0][user_input],"     R= ",user_inputR,"  ",y_predR[0][user_inputR])
                    user_input=np.argmax(y_predL)
                    print(user_inputL,"  ",y_predL[0][user_inputL],"     R= ",user_inputR,"  ",y_predR[0][user_inputR])
                    
                        
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


                            # current_volume = sp.current_playback()['device']['volume_percent']
                            # new_volume = min(current_volume + 10, 100)


                            asyncio.run(change_volume(spotify=sp, volume=100))
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
                            # current_volume = sp.current_playback()['device']['volume_percent']
                            # new_volume = max(current_volume - 10, 0)

                            asyncio.run(change_volume(spotify=sp, volume=50))
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