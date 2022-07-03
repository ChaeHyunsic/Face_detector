import numpy as np, cv2

def preprocessing(no):
    image = cv2.imread('face/%02d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 명암도 영상 변환
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화

    return image, gray

def correct_image(image, face_center, eye_centers):
    pt0, pt1 = eye_centers
    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0

    dx, dy = np.subtract(pt1, pt0).astype(float)
    angle = cv2.fastAtan2(dy, dx)
    rot_mat = cv2.getRotationMatrix2D(face_center, angle, 1)

    size = image.shape[1::-1]
    corr_image = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)

    eye_centers = np.expand_dims(eye_centers, axis=0)            # 차원 증가
    corr_centers = cv2.transform(eye_centers, rot_mat)
    corr_centers = np.squeeze(corr_centers, axis=0)              # 차원 감소

    return corr_image, corr_centers

def define_roi(pt, size):
    return np.ravel([pt, size]).astype(int)

def detect_object(center, face):
    w, h = face[2:4]
    center = np.array(center)
    gap1 = np.multiply((w,h), (0.45, 0.65))     # 평행이동 거리
    gap2 = np.multiply((w,h), (0.20, 0.1))

    pt1 = center - gap1                         # 좌상단 평행이동 - 머리 시작좌표
    pt2 = center + gap1                         # 우하단 평행이동 - 머리 종료좌표
    hair = define_roi(pt1, pt2-pt1)             # 머리 영역

    size = np.multiply(hair[2:4], (1, 0.4))
    hair1 = define_roi(pt1, size)               # 윗머리 영역
    hair2 = define_roi(pt2-size, size)          # 귀밑머리 영역

    lip_center = center + (0, h * 0.3)
    lip1 = lip_center - gap2                    # 좌상단 평행이동
    lip2 = lip_center + gap2                    # 우하단 평행이동
    lip = define_roi(lip1, lip2-lip1)           # 입술 영역

    return [hair1, hair2, lip, hair]

def display(image, centers, sub):
    cv2.circle(image, tuple(centers[0]), 10, (0, 255, 0), 2)    # 눈 표시
    cv2.circle(image, tuple(centers[1]), 10, (0, 255, 0), 2)
    draw_ellipse(image, sub[2], 0.35,(0, 0, 255),  2)	        # 얼굴 표시
    draw_ellipse(image, sub[3], 0.45,(255, 100, 0), 2)          # 입술 표시
    cv2.imshow("FaceDetecter", image)

def draw_ellipse(image, roi, ratio, color, thickness=cv2.FILLED):
    x, y, w, h = roi
    center = (x + w // 2, y + h // 2)
    size = (int(w * ratio), int(h * ratio))
    cv2.ellipse(image, center, size, 0, 0, 360, color, thickness)

    return image

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")   # 얼굴 검출기
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")                 # 눈 검출기

no, max_no, cnt = 0, 60, 1

while True:
    no = no + cnt
    image, gray = preprocessing(no)

    if image is None:
        print("%02d.jpg: 영상 파일 없음" % no)
        if no < 0: no = max_no
        elif no >= max_no: no = 0
        continue

    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
    if faces.any():
        x, y, w, h = faces[0]
        face_image = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

        if len(eyes) == 2:
            face_center = (int(x + w // 2), int(y + h // 2))
            eye_centers = [(x + ex + ew // 2, y + ey + eh // 2) for ex, ey, ew, eh in eyes]

            corr_image, corr_centers = correct_image(image, face_center, eye_centers)  # 기울기 보정
            sub_roi = detect_object(face_center, faces[0])  # 머리 및 입술 영역 검출

            display(corr_image, corr_centers, sub_roi)  # 검출된 영역 표시
        else: print("%02d.jpg: 눈 미검출" % no)
    else: print("%02d.jpg: 얼굴 미검출" % no)

    key = cv2.waitKeyEx(0)                          # 키 이벤트 대기
    if key == 13 or key == 32: cnt = 1              # 엔터 키 이벤트나 스페이스 키 이벤트가 발생하면 다음 영상
    elif key == 8: cnt = -1                         # 백스페이스 키 이벤트가 발생하면 이전 영상
    elif key == 27: break                           # ESC 키 이벤트가 발생하면 종료