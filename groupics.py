#!/usr/bin/env python3
import errno
import sys
import time
from person_db import Person
from person_db import Face
from person_db import PersonDB
import face_recognition
import face_recognition_models
import numpy as np
from datetime import datetime
import cv2
import dlib
import face_alignment_dlib
import os
from PIL import Image
import shutil
import multiprocessing as mp
from multiprocessing import Pool

start_code = time.time()

class InputOrder():

    def __init__(self, list_index):
        self.many_person = []
        self.in_pics = list_index

    def add_pic(self, person_list):
        self.many_person.append(person_list)  #extend


class GrouPics():

    def __init__(self, init_persons):
        self.group = []
        self.group.extend(init_persons)
        self.same_idx = 0
        self.flag_pass = False
        self.add_group = False

    def compare_with_group(self, picture_persons):
        self.same_idx = 0
        temp = []
        self.flag_pass = False
        for picture_person in picture_persons:
            flag = 0
            for group_person in self.group:
                if picture_person == group_person:
                    self.same_idx += 1
                    flag += 1
            if flag == 0:
                temp.append(picture_person)

        if self.same_idx >= 2:
            # add_group_person
            self.group.extend(temp)
            self.add_group = False
            self.flag_pass = True
        else:
            self.add_group = True
            self.flag_pass = False
        temp.clear()

    def check_group(self, other_group):
        self.same_idx = 0
        temp = []
        self.flag_pass = False
        for other_group_person in other_group:
            flag = 0
            for group_person in self.group:
                if other_group_person == group_person:
                    self.same_idx += 1
                    flag += 1
            if flag == 0:
                temp.append(other_group_person)

        # if self.same_idx > min(len(self.group), len(other_group)) * 0.666:
        if self.same_idx >= 4:
            self.group.extend(temp)
            self.add_group = False
            self.flag_pass = True
            # print("합쳐진 그룹:", self.group, "\n비교한 그룹:", other_group)
        elif self.same_idx >= 2 and len(other_group) <= 9 and len(self.group) <= 9:
            self.group.extend(temp)
            self.add_group = False
            self.flag_pass = True
        # elif self.same_idx == 3 and len(other_group) <= 5 and len(self.group) <= 5:
        #     self.group.extend(temp)
        #     self.add_group = False
        #     self.flag_pass = True
        else:
            self.add_group = True
            self.flag_pass = False
        temp.clear()


class OriginImage():
    _last_id = 0

    def __init__(self, filename):
        self.whois = []
        self.many = 0
        self.same_person = 0
        self.group = 0
        self.filename = filename

    def add_he_is(self, person_a):
        # add person_a(whoishe)
        self.whois.append(person_a)
        self.many = len(self.whois)


class FaceClassifier():
    def __init__(self, threshold, ratio):
        self.similarity_threshold = threshold
        self.ratio = ratio
        self.predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())

    def get_face_image(self, frame, box):
        img_height, img_width = frame.shape[:2]
        (box_top, box_right, box_bottom, box_left) = box
        box_width = box_right - box_left
        box_height = box_bottom - box_top
        crop_top = max(box_top - box_height, 0)
        pad_top = -min(box_top - box_height, 0)
        crop_bottom = min(box_bottom + box_height, img_height - 1)
        pad_bottom = max(box_bottom + box_height - img_height, 0)
        crop_left = max(box_left - box_width, 0)
        pad_left = -min(box_left - box_width, 0)
        crop_right = min(box_right + box_width, img_width - 1)
        pad_right = max(box_right + box_width - img_width, 0)
        face_image = frame[crop_top:crop_bottom, crop_left:crop_right]
        if (pad_top == 0 and pad_bottom == 0):
            if (pad_left == 0 and pad_right == 0):
                return face_image  # 성립 조건 만족 얼굴 이미지 리턴
        padded = cv2.copyMakeBorder(face_image, pad_top, pad_bottom,
                                    pad_left, pad_right,
                                    cv2.BORDER_CONSTANT)  # constant 로 이미지 복제 (약간의 액자느낌 색상도 지정 가능해보임)
        return padded

    # return list of dlib.rectangle
    def locate_faces(self, frame):
        # start_time = time.time()
        if self.ratio == 1.0:
            rgb = frame[:, :, ::-1]  # ratio 설정이 default, 즉 1이 들어올 경우 프레임을 이미지로 전환
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)  # ratio 값에 맞게 프레임의 사이즈를 작게 resize
            rgb = small_frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb)  # 얼굴 영역?
        # elapsed_time = time.time() - start_time
        # print("locate_faces takes %.3f seconds" % elapsed_time)
        if self.ratio == 1.0:
            return boxes
        boxes_org_size = []
        for box in boxes:
            (top, right, bottom, left) = box
            left = int(left / ratio)
            right = int(right / ratio)
            top = int(top / ratio)
            bottom = int(bottom / ratio)
            box_org_size = (top, right, bottom, left)
            boxes_org_size.append(box_org_size)
        return boxes_org_size  # ratio 비율에 맞게 박스 수정

    def detect_faces(self, frame, filename):
        boxes = self.locate_faces(frame)  # 박스 locate
        if len(boxes) == 0:
            return []

        # faces found
        faces = []
        # now = datetime.now()  # 필요한가?
        # str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'  # 필요한가?
        str_ms = filename

        for i, box in enumerate(boxes):
            # extract face image from frame
            face_image = self.get_face_image(frame, box)  # frame에서 박스에 있는 얼굴 이미지 추출

            # get aligned image
            aligned_image = face_alignment_dlib.get_aligned_face(self.predictor, face_image)  # 얼굴 (수평) 정렬

            # compute the encoding
            height, width = aligned_image.shape[:2]
            x = int(width / 3)
            y = int(height / 3)
            box_of_face = (y, x * 2, y * 2, x)  # 이 부분은 왜 박스를 이렇게 형성?
            encoding = face_recognition.face_encodings(aligned_image,
                                                       [box_of_face])[0]
            # 파라미터 값을 설정해줄 필요가 있을듯
            face = Face(str_ms + "_" + str(i) + ".png", face_image, encoding)  # 파일이름 지정하여 파일 이름과 함께 얼굴 이미지와 인코딩 데이터를 저장해주는 부분인거같은데 이부분에서 원본도 저장되게 만들수 있나?
            face.location = box  # 얼굴 위치값?
           # cv2.imwrite(str_ms + str(i) + ".r.png", aligned_image)  # cv2를 이용해서 파일 저장
            faces.append(face)  # 얼굴을 추가해주는데 어디를 위해서? 데이터값? 을 위해서인듯
        return faces

    # 분류되어 있는 사람과 비교하는 파트
    def compare_with_known_persons(self, face, persons):
        if len(persons) == 0:
            person = Person()
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            pdb.persons.append(person)
            return person

        # see if the face is a match for the faces of known person
        encodings = [person.encoding for person in persons]  # input persons Used by Kyo
        distances = face_recognition.face_distance(encodings, face.encoding)  # 차이 값 계산(다른 인코딩들과 현재 인코딩값과 거리계산)
        index = np.argmin(distances)  # 차이가 최소가 되는 person의 인덱스를 구한다.
        min_value = distances[index]  # 디스턴스 값을 최소값으로 저장
        if min_value < self.similarity_threshold:  # 최소 디스턴스가 유사도 쓰레스홀드 값보다 작으면
            # face of known person
            persons[index].add_face(face)  # person에 face data 추가
            # re-calculate encoding
            persons[index].calculate_average_encoding()  # 인코딩 평균값 재계산
            face.name = persons[index].name  # person의 name을 face에도 적용
            return persons[index]
        else:
            # Origin UNKNOWN
            person = Person()
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            pdb.persons.append(person)
            return person


    def draw_name(self, frame, face):
        color = (0, 0, 255)  # cv를 이용해서 박스를 drawing
        thickness = 2  # 선 두께
        (top, right, bottom, left) = face.location  # 얼굴 위치

        # draw box
        width = 20  # 박스 폭 설정
        if width > (right - left) // 3:  # 얼굴의 폭이 20보다 작은 경우, 3으로 나눈 후 정수부분만
            width = (right - left) // 3
        height = 20
        if height > (bottom - top) // 3:
            height = (bottom - top) // 3
        cv2.line(frame, (left, top), (left + width, top), color, thickness)
        cv2.line(frame, (right, top), (right - width, top), color, thickness)
        cv2.line(frame, (left, bottom), (left + width, bottom), color, thickness)
        cv2.line(frame, (right, bottom), (right - width, bottom), color, thickness)
        cv2.line(frame, (left, top), (left, top + height), color, thickness)
        cv2.line(frame, (right, top), (right, top + height), color, thickness)
        cv2.line(frame, (left, bottom), (left, bottom - height), color, thickness)
        cv2.line(frame, (right, bottom), (right, bottom - height), color, thickness)

        # draw name
        # cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face.name, (left + 6, bottom + 30), font, 1.0,
                    (255, 255, 255), 1)


class myUtils:
    @staticmethod
    def xgetFileList(dir, ext=('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG',)):
        assert isinstance(ext, tuple)
        matches = []
        for (path, dirnames, files) in os.walk(dir):
            for filename in files:
                if os.path.splitext(filename)[1] in ext:
                    matches.append(os.path.join(path, filename))
        return matches


def replace_foreach(idx, file, _fc):
    print("ck")
    img_RGB = np.array(Image.open(file))
    img = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
    # Image.open(file).show()
    temp_str = file.split(sep="\\")
    temp_str2 = temp_str[-1].split(sep=".")
    faces = _fc.detect_faces(img, temp_str2[0])
    temp_str.clear()
    temp_str2.clear()
    print(faces)
    return (idx, faces)


if __name__ == '__main__':
    print("x")
    import argparse
    import signal
    import time
    import os

    test_number = 330

    ap = argparse.ArgumentParser()
    ap.add_argument("inputfile",
                        help="video file to detect or '0' to detect from web cam")  # 이부분 수정해야하지 않나

    result_dir = 'D:\\face_recognition\\mresult\\recovery_9_' + str(test_number)
    ap.add_argument("-t", "--threshold", default=test_number/1000, type=float,
                        help="threshold of the similarity (default="+str(test_number/1000)+")")
    ap.add_argument("-S", "--seconds", default=1, type=float,
                        help="seconds between capture")
    ap.add_argument("-s", "--stop", default=0, type=int,
                        help="stop detecting after # seconds")
    ap.add_argument("-k", "--skip", default=0, type=int,
                        help="skip detecting for # seconds from the start")
    ap.add_argument("-d", "--display", default=0, action='store_true',
                        help="display the frame in real time")
    ap.add_argument("-c", "--capture", default='cap', type=str,
                        help="save the frames with face in the CAPTURE directory")
    ap.add_argument("-r", "--resize-ratio", default=1.0, type=str,
                        help="resize the frame to process (less time, less accuracy)")
    args = ap.parse_args()

    try:
        if not (os.path.isdir(result_dir)):
            os.makedirs(os.path.join(result_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
        # sys.stdout = open(result_dir + '_output.txt', 'w')

    src_file = args.inputfile
    if src_file == "0":  # 웹캠
        src_file = 0

    src = cv2.VideoCapture(src_file)
    if not src.isOpened():  # 에러코드
        print("cannot open inputfile", src_file)
        exit(1)

    test_path = "D:\\face_recognition\\testimage\\Dongjin"       # testimage\\firstdata
    files = myUtils.xgetFileList(test_path)
    ratio = float(args.resize_ratio)

    if ratio != 1.0:  # 리사이즈 레티오 설정 값 있는 경우
        s = "RESIZE_RATIO: " + args.resize_ratio
        s += " -> %dx%d" % (int(src.get(3) * ratio), int(src.get(4) * ratio))
        print(s)

    # load person DB
    pdb = PersonDB()
    pdb.load_db(result_dir)
    pdb.print_persons()

    fc = FaceClassifier(args.threshold, ratio)
    org = []
    org_img_idx = "global"
    org_img_idx = 0
    max_person_count = 0
    num_capture = 0

    if args.capture:
        print("Captured frames are saved in '%s' directory." % args.capture)
        if not os.path.isdir(args.capture):
            os.mkdir(args.capture)

    process_num = 8
    faces_list = []

    for f in files:
        org.append(OriginImage(f))
    for i, f in enumerate(files):
        faces_list.append(replace_foreach(i, f, fc))
    #     print(i)
    # with Pool(process_num) as p:
    #     faces_list = p.starmap(replace_foreach, [(i, f, fc) for i, f in enumerate(files)])
    for idx, faces in faces_list:
        for face in faces:
            person = fc.compare_with_known_persons(face, pdb.persons)
            org[idx].add_he_is(person.name)
            if org[idx].many > max_person_count:
                max_person_count = org[idx].many
        # print(org[idx].filename, org[idx].whois)  # for test

    input_orders = []
    group_data = []
    for i in range(max_person_count + 1):
        input_orders.append(InputOrder(i))
        idx = 0
        for org_idx in org:
            if org[idx].many is i:
                input_orders[i].add_pic(org[idx].whois)
                print("[", i ,"] ", org[idx].whois)
            idx += 1
        if i == 0:
            continue
        elif i == 1:
            continue
        elif i == 2:
            continue
        # print(i, input_orders[i].many_person)
        for input_idx in range(len(input_orders[i].many_person)):
            if len(group_data) == 0:
                group_data.append(GrouPics(input_orders[i].many_person[input_idx]))
            else:
                for grp_index in range(len(group_data)):
                    group_data[grp_index].compare_with_group(input_orders[i].many_person[input_idx])
                    if group_data[grp_index].flag_pass:
                        print("그룹인원추가: ", input_orders[i].many_person[input_idx], "\n", group_data[grp_index].group)
                        break
                if group_data[grp_index].add_group:
                    group_data.append(GrouPics(input_orders[i].many_person[input_idx]))
                    print("새그룹추가: ", input_orders[i].many_person[input_idx], "\n", group_data[grp_index + 1].group)
        ori_len = 0
        while len(group_data) != ori_len:
            ori_len = len(group_data)
            idx_1 = 0
            for grp_check_idx in group_data:
                idx_2 = 0
                for grp2_check_idx in group_data:
                    if idx_1 >= idx_2:
                        idx_2 += 1
                        continue
                    group_data[idx_1].check_group(group_data[idx_2].group)
                    if group_data[idx_1].flag_pass:
                        if idx_1 != idx_2:
                            group_data.remove(group_data[idx_2])
                            break
                    idx_2 += 1
                idx_1 += 1

    # for idx in range(len(group_data)):
    #     print(idx, "\n", group_data[idx].group)
    # org_img_idx = 0
    # group__idx = 0
    # while (org_img_idx < len(org)):
    #     i = 0
    #     max_set_val = 0
    #     bug_flag = 0
    #     for dat in group_data:
    #         temp = len(set(org[org_img_idx].whois) & set(group_data[i].group))  # 사진 이미지와 그룹데이터비교
    #         if temp > max_set_val:
    #             max_set_val = temp
    #             group__idx = i
    #             bug_flag = 0
    #         elif temp == max_set_val:
    #             bug_flag = 1
    #         i += 1
    #
    #     org[org_img_idx].group = group__idx
    #
    #     print(group__idx, org[org_img_idx].filename , max_set_val, bug_flag)
    #
    #     org_img_idx += 1

    pdb.save_db(result_dir)
    # pdb.print_persons()
    print("time :", time.time() - start_code)