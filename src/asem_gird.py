import cv2
import os
import numpy as np
import json
import re
 
def load_test_images(folder_path , gt_or_test):
    """
    주어진 폴더에서 'test'로 시작하는 이미지 파일들을 numpy 배열로 불러와 리스트에 저장합니다.

    :param folder_path: 이미지가 저장된 폴더의 경로
    :return: 'test'로 시작하는 이미지들의 numpy 배열 리스트
    """
    test_images = []
    # 파일 이름을 숫자 부분에 따라 정렬하기 위해 파일명에서 숫자를 추출하는 함수
    def extract_number(filename):
        return int(filename.split('_')[1].split('.')[0])

    # 지정된 폴더 내의 모든 파일 순회
    for filename in sorted(os.listdir(folder_path), key=extract_number):
        if filename.startswith(gt_or_test) and filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                test_images.append(image)
            else:
                print(f"Failed to load image: {filename}")
    return test_images

def merge_images(grids):
    
   

    top_row = np.concatenate(grids[:2], axis=1)  # 상단 두 이미지 병합
    bottom_row = np.concatenate(grids[2:], axis=1)  # 하단 두 이미지 병합
    merged_image = np.concatenate([top_row, bottom_row], axis=0)  # 두 줄 병합
    return merged_image

def merge_images(grids, n):
    """
    n x n 그리드 이미지를 하나의 큰 이미지로 병합합니다.

    :param grids: n x n 그리드 이미지 리스트
    :param n: 그리드의 수 (세로 및 가로)
    :return: 병합된 이미지
    """
    # 각 행을 생성하기 위해 n개의 이미지를 수평으로 연결합니다.

    rows = [np.concatenate(grids[i*n:(i+1)*n], axis=1) for i in range(n)]
    
    # 모든 행을 수직으로 연결하여 전체 이미지를 만듭니다.
    merged_image = np.concatenate(rows, axis=0)
    return merged_image

def save_merged_images(image_sets, save_folder):
    """
    이미지 세트를 병합하고 지정된 폴더에 저장합니다.

    :param image_sets: 2x2 그리드로 분할된 여러 이미지 세트의 리스트
    :param save_folder: 저장할 폴더 경로
    """
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 이미지 세트를 병합하여 저장
    for idx, grids in enumerate(image_sets):
        merged_image = merge_images(grids)
        save_path = os.path.join(save_folder, f"merged_image_{idx}.png")
        cv2.imwrite(save_path, merged_image)

def load_epoch_info(json_filepath):
    """
    주어진 JSON 파일에서 각 그리드의 epoch 정보를 불러와 2차원 배열로 변환합니다.
    :param json_filepath: JSON 데이터가 저장된 파일 경로
    :return: 각 그리드의 epoch 정보가 담긴 2차원 배열
    """
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # 데이터에서 그리드 크기를 추정합니다.
    grid_size = int(np.sqrt(len(data) * len(data[0])))
    epoch_info = np.zeros((grid_size, grid_size), dtype=int)

    for i, row in enumerate(data):
        for item in row:
            index_i = item['index_i']
            index_j = item['index_j']
            epoch = item['epoch']
            epoch_info[index_i, index_j] = epoch

    return epoch_info

def make_result_with_json(n,d,w,namse_base ,json_filepath , result_path):
    
    imgss = []
    result = []
    paths  = []
    out_path = os.path.join("result_json_img",result_path)
    out_path = os.path.join(out_path, "redering_result")


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    
    model_name = f"knight{n}_lr_d{d}w{w}"
    epochs = load_epoch_info(json_filepath)
    print("test")

    


    for i in range(n):
        for j in range(n):
            str = f"result/Exp_uvsv_{namse_base}_{model_name}_i{i}_j{j}/test/epoch-{epochs[i][j]}"
            # 240119_knight32_lr_d4w32_i${i}_j${j}
            paths.append(str) 

    for path in paths:
        imgss.append(load_test_images(path , "test"))

    imgss = np.transpose(imgss,(1,0,2,3,4))
    for imgs in imgss:

        result.append(merge_images(imgs , n))


                  
    print("test")
      # 실제 저장할 경로로 변경해야 합니다.

    # 저장할 폴더가 없으면 생성
    
    # result 리스트의 각 이미지를 저장
    for idx, image in enumerate(result):
        # 저장할 이미지 파일 경로
        save_path = os.path.join(out_path, f'merged_image_{idx}.png')
        # 이미지 저장
        cv2.imwrite(save_path, image)        

def make_result(n,d,w,epoch):
    
    imgss = []
    result = []
    paths  = []
    test_result_path = "m_test_result"
    
    model_name = f"knight{n}_lr_d{d}w{w}"
    epoch_num = epoch

    for i in range(n):
        for j in range(n):
            str = f"result/Exp_uvsv_240119_{model_name}_i{i}_j{j}/test/epoch-{epoch_num}"
            # 240119_knight32_lr_d4w32_i${i}_j${j}
            paths.append(str) 

    for path in paths:
        imgss.append(load_test_images(path , "test"))

    imgss = np.transpose(imgss,(1,0,2,3,4))
    for imgs in imgss:

        result.append(merge_images(imgs , n))


                  
    print("test")
    save_folder = os.path.join(test_result_path, f"{model_name}_epoch{epoch_num}")  # 실제 저장할 경로로 변경해야 합니다.

    # 저장할 폴더가 없으면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # result 리스트의 각 이미지를 저장
    for idx, image in enumerate(result):
        # 저장할 이미지 파일 경로
        save_path = os.path.join(save_folder, f'merged_image_{idx}.png')
        # 이미지 저장
        cv2.imwrite(save_path, image)
def parse_string(input_string):
    # '_knight' 앞에 오는 문자열 추출
    prefix = re.search(r'(.*)_knight', input_string).group(1)

    # 'knight' 다음에 오는 숫자 추출
    knight_number = int(re.search(r'knight(\d+)', input_string).group(1))

    # 'd' 다음에 오는 숫자 추출
    d_number = int(re.search(r'd(\d+)', input_string).group(1))

    # 'w' 다음에 오는 숫자 추출
    w_number = int(re.search(r'w(\d+)', input_string).group(1))

    return prefix, knight_number, d_number, w_number

def make_result_with_json2(file_name):
    in_path = os.path.join("result_json",f"{file_name}.json")
    base_name ,grid_n , d, w  = parse_string(file_name)
    make_result_with_json(grid_n,d,w,base_name,in_path,file_name)




if __name__ =="__main__":
    #make_result_with_json2("2_test_lossclamp_knight16_lr_d4w32")


    epochs = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
    for epoch in epochs:
        make_result(8,2,16,epoch)
    #make_result_with_json(32, 4, 64,  "2_test_lossclamp" ,'result_json/2_test_lossclamp_knight32_lr_d4w64.json')
    # make_result_with_json(16, 4, 32,  "2_test_lossclamp" ,'result_json/2_test_lossclamp_knight16_lr_d4w32.json')

    
    # make_result_with_json(32, 4, 16, "2_test_lossclamp" , 'result_json/2_test_lossclamp_knight32_lr_d4w16.json')
    # make_result_with_json(32, 4, 32, "2_test_lossclamp" , 'result_json/2_test_lossclamp_knight32_lr_d4w32.json')

    #1_test_lossclamp_knight16_lr_d4w64_i${i}_j${j}