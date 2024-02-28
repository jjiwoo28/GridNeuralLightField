import cv2
import os
import numpy as np
import json

def load_test_images(folder_path, gt_or_test):
    """
    주어진 폴더에서 'test'로 시작하는 이미지 파일들을 numpy 배열로 불러와 리스트에 저장합니다.

    :param folder_path: 이미지가 저장된 폴더의 경로
    :param gt_or_test: 이미지 파일 이름이 시작하는 문자열 ('test' 또는 'gt')
    :return: 'test' 또는 'gt'로 시작하는 이미지들의 numpy 배열 리스트
    """
    images = []
    for filename in sorted(os.listdir(folder_path), key=lambda f: int(f.split('_')[1].split('.')[0])):
        if filename.startswith(gt_or_test) and filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
            else:
                print(f"Failed to load image: {filename}")
    return np.array(images)

def merge_images(images, n):
    """
    n x n 그리드 이미지를 하나의 큰 이미지로 병합합니다.

    :param images: n x n 그리드 이미지 리스트
    :param n: 그리드의 수 (세로 및 가로)
    :return: 병합된 이미지
    """
    
    rows = [np.concatenate(images[i*n:(i+1)*n], axis=1) for i in range(n)]
    return np.concatenate(rows, axis=0)

def make_result(n, d, w , json_filepath):
    """
    지정된 그리드 크기, 네트워크 깊이, 네트워크 너비에 따라 결과 이미지를 생성합니다.

    :param n: 그리드 크기
    :param d: 네트워크 깊이
    :param w: 네트워크 너비
    :param epoch: 에폭 정보가 담긴 JSON 파일 경로
    :param json_filepath: JSON 파일 경로
    """
    # JSON 파일에서 에폭 정보 로드
    with open(json_filepath, 'r') as file:
        epoch_data = json.load(file)

    test_result_path = "m_test_result"
    model_name = f"knight{n}_lr_d{d}w{w}"

    for idx, row in enumerate(epoch_data):
        for idy, record in enumerate(row):
            #1_test_lossclamp_knight16_lr_d4w64_
            folder_path = f"result/Exp_uvsv_2_test_lossclamp_{model_name}_i{record['index_i']}_j{record['index_j']}/test/epoch-{record['epoch']}"
            images = load_test_images(folder_path, "test")
            merged_image = merge_images(images, n)

            # 저장할 폴더 생성
            save_folder = os.path.join(test_result_path, f"{model_name}_epoch{record['epoch']}")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 이미지 저장
            save_path = os.path.join(save_folder, f"merged_image_{idx}_{idy}.png")
            cv2.imwrite(save_path, merged_image)

if __name__ == "__main__":
    
    make_result(16, 4, 32,  'result_json/2_test_lossclamp_knight16_lr_d4w32.json')

    
    # make_result(32, 4, 16,  'result_json/2_test_lossclamp_knight32_lr_d4w16.json')
    # make_result(32, 4, 32,  'result_json/2_test_lossclamp_knight32_lr_d4w32.json')
