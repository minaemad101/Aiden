import cv2
import mediapipe as mp
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="7b25cd62f12b433aaafd86c1fdd2b81c",
                                               client_secret="eaf42cb2254e430faddd54232664daad",
                                               redirect_uri="http://google.com/callback/",
                                               scope="user-read-playback-state user-modify-playback-state"))
devices = sp.devices()
print(devices)
device_id = devices['devices'][0]['id']
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
                        playback = sp.current_playback()
                        if playback and playback['is_playing']:
                            sp.pause_playback()
                        else:
                            print("A song is currently stopping.")
                        s=time.time()
                        count=0
                        temp=True
                        print('Song is start stoping.')
                    if i==0 :
                        count+=1
                        i=0
                    else:
                        count=1
                        i=0    
                elif user_input == 1:
                    if count==10 and i==1:
                        sp.previous_track(device_id=device_id)
                        count=0
                        print('Song is pre playing.')
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
                        sp.volume(new_volume, device_id=device_id)
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
                        sp.volume(new_volume, device_id=device_id)
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
                        sp.next_track(device_id=device_id)
                        count=0
                        s=time.time()
                        print('Song is next playing.')
                        temp=True

                            
                    if i==4 :
                        count+=1
                        i=4
                    else:
                        count=1
                        i=4
                
                elif user_input == 5:
                    if count==10 and i==5:
                        playback = sp.current_playback()
                        if playback and playback['is_playing']:
                            print("A song is currently playing.")
                        else:
                            sp.start_playback(device_id=device_id)
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

        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

cap.release()
# cv2.destroyAllWindows()