from cvzone.HandTrackingModule import HandDetector
import cv2
        
detector = HandDetector(maxHands = 1) # 손 인식 모델 초기화(maxHands -> 인식할 손 개수)

cap_cam = cv2.VideoCapture(0)
cap_video = cv2.VideoCapture('video.mp4') # mp4 파일 불러오기

w = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # cap_cam의 가로 길이 불러오기

total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT)) # 비디오의 프레임 개수
print(total_frames) # 250개 

_, video_img = cap_video.read() # 첫 번쨰 프레임을 읽는다.

def draw_timeline(video_img, rel_x):
    img_h, img_w, img_c = video_img.shape
    timeline_w = int(img_w * rel_x)
    cv2.rectangle(video_img, pt1=(0, img_h - 50), pt2=(timeline_w, img_h - 45), color=(0, 0, 255), thickness=-1)

rel_x = 0
frame_idx = 0
draw_timeline(video_img, rel_x)

while cap_cam.isOpened():
    ret, cam_img = cap_cam.read()

    if not ret: break

    cam_img = cv2.flip(cam_img, 1) # 캠 flip

    hands, cam_img = detector.findHands(cam_img) # cvzone을 사용해서 손 인식하기

    if hands:
        lm_list= hands[0]['lmList'] 
        # 손의 영역들을 출력해준다. mediapipe랑 다른 점은 mediapipe는 0에서 1 사이의 값으로 출력했다면 cvzone은 픽셀 좌표로 변환이 돼서 저장됨.

        fingers = detector.fingersUp(hands[0]) # 손을 구부렸는지 구부리지 않았는지 확인(손가락을 접으면 0, 펴면 1)
        #cv2.putText(cam_img, text=str(fingers), org=(100, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=3)

        length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img) # 손 랜드마크의 4, 8번째(각각 엄지, 검지) 사이의 거리를 구하고 cam_img에 그림
        # cvzone 1.5.6 버전을 사용할 땐 too many values to unpack (expected 2) 오류가 떴었는데, 이를 1.5.4 버전으로 다운그레이드 하니 해결되긴 했다.
        # 왜 그런지는 모름.. 버전 업데이트 하면서 뭐가 바꼈나..?
        
        if fingers == [0, 0, 0, 0, 0]: # 정지모드
            pass
        else: # 탐색 or 플레이 모드
            if length < 50 : #Navigate 탐색 -> 만약 length가 20 이하라면 어떤 기능을 수행
                rel_x = lm_list[4][0] / w # 엄지손가락의 x 좌표 구하기(상대좌표(0-1) : 손가락 위치의 좌표를 캠의 width 길이로 나눈다.)
                
                frame_idx = int(rel_x * total_frames) # 프레임 번호 구하기(rel_x가 0과 가까울 땐 프레임도 0과 가까울 것이고, rel_x가 1과 가까워 질 땐 프레임은 250에 가까워진다.)
                if frame_idx < 0 : frame_idx = 0
                elif frame_idx > total_frames : frame_idx = total_frames    

                cap_video.set(1, frame_idx) # 동영상을 해당 프레임(frame_idx)으로 이동
                
                cv2.putText(cam_img, text='Navigate %.2f, %d' % (rel_x, frame_idx), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=3)
                
            else : #Play 재생
                frame_idx = frame_idx + 1
                rel_x = frame_idx/ total_frames

            if frame_idx < total_frames:
                _, video_img = cap_video.read() # 동영상의 프레임을 읽는다.
                draw_timeline(video_img, rel_x)

    cv2.imshow('cam', cam_img)
    cv2.imshow('video', video_img)
    if cv2.waitKey(1) == ord('q'):
        break

   