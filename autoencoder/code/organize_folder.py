#누끼처리된 데이터
import os
import shutil

data_path = './workspace_project/9oz/A/17'  # 이미지가 있는 폴더 경로
output_path = './workspace_project/9oz/A/17_sorted'  # 클래스별로 정리된 폴더가 생성될 경로


# 클래스별로 폴더를 생성하고 이미지를 해당 폴더로 이동하는 함수
def organize_images(data_path, output_path):
    os.makedirs(output_path, exist_ok=True)  # 출력 폴더 생성

    for filename in os.listdir(data_path):
        if filename.endswith('.out.jpg'):  # 확장자가 .out.jpg인 경우에만 처리(누끼처리됨)
            class_name = filename[4:6]     # 복종코드 추출
            class_path = os.path.join(output_path, class_name)

            os.makedirs(class_path, exist_ok=True)  # 클래스별 폴더 생성

            old_path = os.path.join(data_path, filename)
            new_path = os.path.join(class_path, filename)

            shutil.move(old_path, new_path)  # 이미지 이동

# 이미지 정리 수행
organize_images(data_path, output_path)