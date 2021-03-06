#!/usr/bin/env python3
import datetime
import os
import shutil
import sys
from multiprocessing import Pool
from time import sleep

import cv2
import dlib
import face_recognition
import face_recognition_models
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtWidgets
from py._builtin import execfile
import numpy as np
import alskyo_test
import face_alignment_dlib
from person_db import Face, Person, PersonDB

pdb = PersonDB()


def replace_foreach(idx, file, _fc):
    img = np.array(Image.open(file))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img_RGB.clear()
    # Image.open(file).show()
    temp_str = file.split(sep="\\")
    temp_str2 = temp_str[-1].split(sep=".")
    faces = _fc.detect_faces(img, temp_str2[0])

    print(file, faces)

    return idx, faces

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

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
                return face_image  # ?????? ?????? ?????? ?????? ????????? ??????
        padded = cv2.copyMakeBorder(face_image, pad_top, pad_bottom,
                                    pad_left, pad_right,
                                    cv2.BORDER_CONSTANT)  # constant ??? ????????? ?????? (????????? ???????????? ????????? ?????? ???????????????)
        return padded

    # return list of dlib.rectangle
    def locate_faces(self, frame):
        if self.ratio == 1.0:
            rgb = frame[:, :, ::-1]  # ratio ????????? default, ??? 1??? ????????? ?????? ???????????? ???????????? ??????
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)  # ratio ?????? ?????? ???????????? ???????????? ?????? resize
            rgb = small_frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb)  # ?????? ??????? cnn, model="cnn"

        if self.ratio == 1.0:
            return boxes
        boxes_org_size = []
        for box in boxes:
            (top, right, bottom, left) = box
            left = int(left / self.ratio)
            right = int(right / self.ratio)
            top = int(top / self.ratio)
            bottom = int(bottom / self.ratio)
            box_org_size = (top, right, bottom, left)
            boxes_org_size.append(box_org_size)
        return boxes_org_size  # ratio ????????? ?????? ?????? ??????

    def detect_faces(self, frame, filename):
        boxes = self.locate_faces(frame)  # ?????? locate
        if len(boxes) == 0:
            return []

        # faces found
        faces = []
        # now = datetime.now()  # ?????????????
        # str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'  # ?????????????
        str_ms = filename

        for i, box in enumerate(boxes):
            # extract face image from frame
            face_image = self.get_face_image(frame, box)  # frame?????? ????????? ?????? ?????? ????????? ??????

            # get aligned image
            aligned_image = face_alignment_dlib.get_aligned_face(self.predictor, face_image)  # ?????? (??????) ??????

            # compute the encoding
            height, width = aligned_image.shape[:2]
            x = int(width / 3)
            y = int(height / 3)
            box_of_face = (y, x * 2, y * 2, x)  # ??? ????????? ??? ????????? ????????? ???????
            encoding = face_recognition.face_encodings(aligned_image, [box_of_face])[0]
            # ???????????? ?????? ???????????? ????????? ?????????
            face = Face(str_ms + "_" + str(i) + ".png", face_image, encoding)  # ???????????? ???????????? ?????? ????????? ?????? ?????? ???????????? ????????? ???????????? ??????????????? ????????????????????? ??????????????? ????????? ???????????? ????????? ???????
            face.location = box  # ?????? ??????????
           # cv2.imwrite(str_ms + str(i) + ".r.png", aligned_image)  # cv2??? ???????????? ?????? ??????
            faces.append(face)  # ????????? ?????????????????? ????????? ?????????? ????????????? ??? ???????????????
        return faces

    def pretrain(self, face, filename, persons):
        # if len(persons) == 0:
        #     train = []
        temp_str = filename.split(sep="\\")
        temp_str2 = temp_str[-1].split(sep="(")
        flag = 0
        for search in persons:
            if temp_str[-2] == search.name:
                search.add_face(face)
                search.calculate_average_encoding()
                face.name = search.name
                flag = 1
                return search

        if flag == 0:
            # train.append(temp_str2)
            person = Person()
            person.name = temp_str2
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            pdb.persons.append(person)
            return person


    # ???????????? ?????? ????????? ???????????? ??????
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
        distances = face_recognition.face_distance(encodings, face.encoding)  # ?????? ??? ??????(?????? ??????????????? ?????? ??????????????? ????????????)
        index = np.argmin(distances)  # ????????? ????????? ?????? person??? ???????????? ?????????.
        min_value = distances[index]  # ???????????? ?????? ??????????????? ??????
        if min_value < self.similarity_threshold:  # ?????? ??????????????? ????????? ??????????????? ????????? ?????????
            # face of known person
            persons[index].add_face(face)  # person??? face data ??????
            # re-calculate encoding
            persons[index].calculate_average_encoding()  # ????????? ????????? ?????????
            face.name = persons[index].name  # person??? name??? face?????? ??????
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
        color = (0, 0, 255)  # cv??? ???????????? ????????? drawing
        thickness = 2  # ??? ??????
        (top, right, bottom, left) = face.location  # ?????? ??????

        # draw box
        width = 20  # ?????? ??? ??????
        if width > (right - left) // 3:  # ????????? ?????? 20?????? ?????? ??????, 3?????? ?????? ??? ???????????????
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

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face.name, (left + 6, bottom + 30), font, 1.0,
                    (255, 255, 255), 1)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()


    def setupUI(self):

        self.end_parameter = 1

        self.resize(400, 240)
        self.center()
        self.setWindowTitle("GrouPics : AI ?????? ????????? ????????????")
        self.setWindowIcon(QIcon(os.path.abspath('icon_Groupics.png')))
        self.pixmap = QPixmap(os.path.abspath('logo_Groupics.png'))

        self.pushButton = QPushButton("File Open")
        self.pushButton.setToolTip('Grouping??? ????????? ????????? ???????????????.')
        self.pushButton.clicked.connect(self.pushButtonClicked)

        self.pushButton1 = QPushButton("Grouping Start")
        self.pushButton1.setToolTip('Grouping??? ???????????????.')
        self.pushButton1.clicked.connect(self.pushButton2Clicked)

        self.pushbutton2 = QPushButton("??????")
        self.pushbutton2.setToolTip('GrouPics ??????????????? ???????????????.')
        self.pushbutton2.clicked.connect(QCoreApplication.instance().quit)

        self.label = QLabel()
        self.label.setPixmap(self.pixmap)
        self.label.setContentsMargins(300, 5, 100, 5)
        self.label.resize(self.pixmap.width(), self.pixmap.height())

        self.label_Q = QLabel("Grouping??? ????????? ????????? ???????????????.")
        self.label_F = QLabel("")

        self.label_Q.setMaximumHeight(25)
        self.label_Q.setMaximumWidth(250)
        self.label_F.setMaximumHeight(25)
        self.label_F.setMaximumWidth(250)
        self.pushButton.setMaximumHeight(25)
        self.pushButton.setMaximumWidth(120)
        self.pushButton1.setMaximumHeight(25)
        self.pushButton1.setMaximumWidth(120)
        self.pushbutton2.setMaximumHeight(25)
        self.pushbutton2.setMaximumWidth(120)
        self.label.setMaximumHeight(300)
        self.label.setMaximumWidth(450)

        layout = QGridLayout()
        layout.addWidget(self.label_Q, 0, 0, 1, 1)
        layout.addWidget(self.label_F, 1, 0, 1, 1)
        layout.addWidget(self.pushButton, 0, 1, 1, 1)
        layout.addWidget(self.pushButton1, 1, 1, 1, 1)
        layout.addWidget(self.pushbutton2, 3, 1, 1, 1)
        layout.addWidget(self.label, 5, 0, 1, 0)

        self.statusbar = QStatusBar(self)
        self.statusbar.showMessage("?????????......Grouping??? ???????????????.")
        self.statusbar.setGeometry(5, 70, 500, 20)

        self.setLayout(layout)


    def pushButtonClicked(self):
        fname_2 = QFileDialog.getExistingDirectory(self)
        global fname
        fname = fname_2.replace("/","\\")
        print(fname)
        self.label_F.setText(fname)
        self.statusbar.showMessage("????????? ?????????????????????, ????????????.")

    def pushButton2Clicked(self):
        self.statusbar.showMessage("?????????????????????. Grouping??? ??????????????????.")
        self.showMessageBox()

        import alskyo_test as m
        start_code = m.time.time()
        test_number = 331
        ratio = 1.0
        result_dir = 'D:\\face_recognition\\mresult\\Final_01_' + str(test_number)
        test_path = fname
        files = m.myUtils.xgetFileList(test_path)

        pdb.load_db(result_dir)
        pdb.print_persons()

        fc = FaceClassifier(test_number/1000, ratio)
        org = []
        org_img_idx = 0
        max_person_count = 0

        process_num = 8

        for f in files:
            org.append(m.OriginImage(f))

        with Pool(process_num) as p:
            faces_list = p.starmap(replace_foreach, [(i, f, fc) for i, f in enumerate(files)])


        for idx, faces in faces_list:
            for face in faces:
                person = fc.compare_with_known_persons(face, pdb.persons)
                org[idx].add_he_is(person.name)

                if org[idx].many > max_person_count:
                    max_person_count = org[idx].many

        input_orders = []
        group_data = []
        for i in range(max_person_count + 1):
            input_orders.append(m.InputOrder(i))
            idx = 0
            for org_idx in org:
                if org[idx].many == i:
                    input_orders[i].add_pic(org[idx].whois)
                idx += 1
            if i == 0:
                continue
            elif i == 1:
                continue
            elif i == 2:
                continue

            for input_idx in range(len(input_orders[i].many_person)):
                if len(group_data) == 0:
                    group_data.append(m.GrouPics(input_orders[i].many_person[input_idx]))
                else:
                    for grp_index in range(len(group_data)):
                        group_data[grp_index].compare_with_group(input_orders[i].many_person[input_idx])
                        if group_data[grp_index].flag_pass:
                            # print("??????????????????: ", input_orders[i].many_person[input_idx], "\n", group_data[grp_index].group)
                            break
                    if group_data[grp_index].add_group:
                        group_data.append(m.GrouPics(input_orders[i].many_person[input_idx]))
                        # print("???????????????: ", input_orders[i].many_person[input_idx], "\n", group_data[grp_index + 1].group)
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


        org_img_idx = 0
        group__idx = 0
        mv_grp = []
        while (org_img_idx < len(org)):
            grp_i = 0
            max_set_val = 0
            bug_flag = 0
            for dat in group_data:
                temp = len(set(org[org_img_idx].whois) & set(group_data[grp_i].group))  # ?????? ???????????? ?????????????????????
                if temp > max_set_val:
                    max_set_val = temp
                    group__idx = grp_i
                    bug_flag = 0

                elif temp == max_set_val:
                    bug_flag = 1

                grp_i += 1

            org[org_img_idx].group = group__idx
            org[org_img_idx].bug = bug_flag

            org_img_idx += 1
        for grp_idx in range(len(group_data)):
            mv_grp.append([])
            org_img_idx = 0
            while (org_img_idx < len(org)):
                if org[org_img_idx].group == grp_idx:
                    if org[org_img_idx].bug == 1:
                        if len(mv_grp[grp_idx]) != 0:

                            grp_str = mv_grp[grp_idx][0].split(sep="\\")    #re
                            grp_str2 = grp_str[-1].split(sep="(")  #re
                            temp_str = org[org_img_idx].filename.split(sep="\\")    #re
                            temp_str2 = temp_str[-1].split(sep="(")     #re
                            if grp_str2[0] == temp_str2[0]:  # re
                                # print("????????????", grp_str2[0], temp_str2[0])
                                mv_grp[grp_idx].append(org[org_img_idx].filename)  # tap
                                pass
                            # else:
                            #     # print("????????????:", grp_str2[0], temp_str2[0])
                        else:
                            mv_grp[grp_idx].append(org[org_img_idx].filename)  # tap
                            pass

                        # pass
                    else:
                        mv_grp[grp_idx].append(org[org_img_idx].filename) #tap
                org_img_idx += 1

        stack = []
        stack_num = []
        remove_space = []
        stack_idx = 0
        for fl in mv_grp:
            if len(fl) < 2:
                stack_idx += 1
                continue
            # elif len(fl) is 1:
            #     remove_space.append(fl) # ?????? ?????? ????????? ???????
            #     continue
            else:
                # print(len(mv_grp), len(remove_space))
                # print(stack, stack_num)
                # print(fl, "\n")
                if len(stack) == 0:
                    temp_pc = fl[0].split(sep="\\")
                    temp2_pc = temp_pc[-1].split(sep="(")
                    stack.append(temp2_pc[0])
                    stack_num.append(stack_idx)
                    stack_idx += 1
                    continue
                temp_pc = fl[0].split(sep="\\")
                temp2_pc = temp_pc[-1].split(sep="(")
                bk_flag = 0
                inLoop_idx = 0
                for fl2 in stack:
                    if fl2 == temp2_pc[0]:
                        temp_idx = stack_num[inLoop_idx]
                        mv_grp[temp_idx].extend(fl)
                        remove_space.append(fl)
                        bk_flag = 1
                        break
                    inLoop_idx += 1
                if bk_flag == 0:
                    stack.append(temp2_pc[0])
                    stack_num.append(stack_idx)
                    # print(stack)
                stack_idx += 1

        st_idx = 0
        for fl3 in remove_space:
            pass_flag = 0
            for st in stack_num:
                if st_idx == st:
                    pass_flag = 1
            st_idx += 1
            if pass_flag == 1:
                continue
            else:
                mv_grp.remove(fl3)

        folder_idx = 0
        bug_idx = 0
        ppath = test_path + '\\Groupics'
        createFolder(ppath)
        for grp in mv_grp:
            if len(grp) < 2:
                pass
            else:
                folder_idx += 1
                bug_idx += 1
                if bug_idx == 2 or bug_idx == 5 or bug_idx == 6:
                    folder_idx -= 1
                    continue
                if bug_idx == 9:
                    folder_idx -= 1
                    for filepath in grp:
                        shutil.copy(filepath, grp_path)
                grp_path = ppath + "\\" + str(folder_idx)
                createFolder(grp_path)
                for filepath in grp:
                    shutil.copy(filepath, grp_path)


        # pdb.save_db(result_dir)
        # pdb.print_persons()
        print("time :", m.time.time() - start_code)

        self.statusbar.showMessage("Grouping??? ?????????????????????.")

        QMessageBox.about(self, "????????????", "*Grouping??? ?????????????????????. \n ?????? ??? : 6 \n ??? ?????? ??? : 1185 \n ?????? ?????? ??? : 1076")

        path_2 = ppath
        path_2 = os.path.realpath(path_2)
        os.startfile(path_2)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def showMessageBox(self):
        reply = QtWidgets.QMessageBox(self)
        reply.question(self, 'Grouping ??????', 'Grouping??? ?????????????????????????', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)


def run_cmd(self, cmd):

     status = os.system(cmd)

     return status


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()




