import os
import re
import cv2
import numpy as np
import subprocess

import flowiz as fz
from PIL import Image




def parse_filename_and_store(file_name, array):
    """
    Parse the filename to extract two numbers and store the filename in the array at the corresponding index.
    """
    # Extract numbers from the filename using regular expression
    match = re.search(r"out_(\d+)_(\d+)_", file_name)
    if match:
        # Convert the extracted numbers to integers
        i, j = int(match.group(1)), int(match.group(2))
        # Store the filename in the array
        array[i][j] = file_name
    return array
def process_directory_path(folder_path):
    """
    Process the files in a given directory, assuming they are in the specified format.
    """
    file_array = [["" for _ in range(100)] for _ in range(100)]  # 100x100 크기의 2차원 배열 초기화

    # 폴더 내의 모든 파일 이름을 가져온다
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)

        # 파일이 일반 파일인지 확인한다
        if os.path.isfile(full_path):
            file_array = parse_filename_and_store(file_name, file_array)

    return file_array

def copy_to_smaller_array(original_array):
    max_index = 0

    # 원본 배열을 순회하여 가장 큰 인덱스 찾기
    for i in range(len(original_array)):
        for j in range(len(original_array[i])):
            if original_array[i][j] != "":
                max_index = max(max_index, i, j)

    # 새 배열 생성 (max_index + 1 크기)
    new_array_size = max_index + 1
    new_array = [["" for _ in range(new_array_size)] for _ in range(new_array_size)]

    # 데이터 복사
    for i in range(new_array_size):
        for j in range(new_array_size):
            new_array[i][j] = original_array[i][j]

    return new_array


def calculate_optical_flow(image1_path, image2_path):
    
    
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    """
    두 그레이스케일 이미지 간의 옵티컬 플로우를 계산합니다.

    :param image1: 첫 번째 이미지 (그레이스케일)
    :param image2: 두 번째 이미지 (그레이스케일)
    :return: 옵티컬 플로우 벡터 필드
    """
    # Farneback 방법을 사용하여 옵티컬 플로우 계산
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def visualize_optical_flow(flow):
    """
    옵티컬 플로우 벡터를 이미지로 시각화합니다.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hue = angle * 180 / np.pi / 2
    saturation = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    value = np.ones_like(hue) * 255
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def save_flow_image(flow, file_path):
    """
    옵티컬 플로우 이미지를 파일로 저장합니다.
    """
    flow_image = visualize_optical_flow(flow)
    cv2.imwrite(file_path, flow_image)

def save_flow_as_image(flow, file_path):
    """
    옵티컬 플로우의 크기에 따라 색상을 달리하여 이미지로 저장합니다.
    
    :param flow: 1차원 옵티컬 플로우 배열 (flow_h 또는 flow_v)
    :param file_path: 저장할 이미지 파일의 경로
    """
    # 옵티컬 플로우의 절대값을 취하여 크기만 고려
    #abs_flow = np.abs(flow)

    abs_flow = flow

    # 크기를 0-255 범위로 정규화
    normalized_flow = cv2.normalize(abs_flow, None, 0, 255, cv2.NORM_MINMAX)

    # uint8 타입으로 변환
    normalized_flow = normalized_flow.astype(np.uint8)

    # 이미지 파일로 저장
    cv2.imwrite(file_path + ".png", normalized_flow)

    np.save(file_path + ".npy",flow)

def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")

def run_test():
    dataset_path = 'dataset/knight'
    img_path = os.path.join(dataset_path,"images")
    array = copy_to_smaller_array(process_directory_path(img_path))
    idx =1
    for i in range(len(array)):
        for j in range(len(array[i])):
            if i ==len(array)-1 or j == len(array[i])-1:
                index_h = j+1
                index_v = i+1
                print(array[i][j])
                if i ==len(array)-1:
                    index_v = i-1
                if j == len(array[i])-1:
                    index_h = j-1
                    
                img = os.path.join(img_path,array[i][j])
                img_h = os.path.join(img_path,array[i][index_h])
                img_v = os.path.join(img_path,array[index_v][j])
                if i ==len(array)-1:
                    flow_h = deepflow2(img,img_h,dataset_path)
                    flow_v = deepflow2(img_v,img,dataset_path)
                elif j == len(array[i])-1:
                    flow_h = deepflow2(img_h,img,dataset_path)
                    flow_v = deepflow2(img,img_v,dataset_path)
                else:
                    flow_h = deepflow2(img,img_h,dataset_path)
                    flow_v = deepflow2(img,img_v,dataset_path)
                flow_h2 = flow_h[:,:,0]
                flow_v2 = flow_v[:,:,1]
                
            
                flow_h2_path = os.path.join(dataset_path,"flow_h_deepflow2")
                create_directory(flow_h2_path)
                save_flow_as_image(flow_h2,os.path.join(flow_h2_path,f"flow_h_i-{i:02d}_j-{j:02d}"))

                flow_v2_path = os.path.join(dataset_path,"flow_v_deepflow2")
                create_directory(flow_v2_path)
                save_flow_as_image(flow_v2,os.path.join(flow_v2_path,f"flow_v_i-{i:02d}_j-{j:02d}"))

            idx+=1
            
   
            # flow_h_path = os.path.join(dataset_path,"flow_h")
            # create_directory(flow_h_path)
            # save_flow_image(flow_h,os.path.join(flow_h_path,f"flow_h_i-{i}_j-{j}.png"))
            
            # flow_v_path = os.path.join(dataset_path,"flow_v")
            # create_directory(flow_v_path)
            # save_flow_image(flow_v,os.path.join(flow_v_path,f"flow_v_i-{i}_j-{j}.png"))
            
            
                            
    

    
    print("test")

def rename(directory_path):

    for filename in os.listdir(directory_path):

    # 정규 표현식으로 i와 j 뒤에 오는 한 자리 숫자를 찾아 두 자리 숫자로 변환

        new_filename = re.sub(r'(?<=i-)\d(?=\D)|(?<=j-)\d(?=\D)', r'0\g<0>', filename)

        if new_filename != filename:  # 파일 이름이 변경되었으면

            old_file_path = os.path.join(directory_path, filename)  # 이전 파일 경로

            new_file_path = os.path.join(directory_path, new_filename)  # 새 파일 경로

            os.rename(old_file_path, new_file_path)  # 파일 이름 변경
        
    print("tesddffs")





def deepflow2(img1_path, img2_path, base_path, options=""):
    output_path = os.path.join(base_path, "output.flo")
    deepflow_path = os.path.join(base_path,"deepflow2")
    create_directory(deepflow_path)
    
    
    
    command = ["../DeepFlow_release2.0/deepflow2", img1_path, img2_path, output_path] + options.split()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    output_array = fz.read_flow(output_path)

    return output_array
    


def run_deepflow2(img1_path, img2_path, output_path, options=""):
    command = ["../DeepFlow_release2.0/deepflow2", img1_path, img2_path, output_path] + options.split()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    print("240107_2222")
    run_test()

    # dataset_path = 'dataset/knight'
    # path1 = os.path.join(dataset_path,"flow_h_deepflow2")
    # path2 = os.path.join(dataset_path,"flow_v_deepflow2")

    # rename(path1)
    # rename(path2)
    
    # dataset_path = 'dataset/knight'
    # img_path = os.path.join(dataset_path,"images")
    # array = copy_to_smaller_array(process_directory_path(img_path))
    # i = 0
    # j = 0
    # index_h = j+1
    # index_v = i+1

    # img = os.path.join(img_path,array[i][j])
    # img_h = os.path.join(img_path,array[i][index_h])
    # img_v = os.path.join(img_path,array[index_v][j])

    # deepflow2_path = os.path.join(dataset_path,"deepflow2")
    # create_directory(deepflow2_path)
    # out_v = os.path.join(deepflow2_path,"output_v.flo")
    # out_h = os.path.join(deepflow2_path,"output_h.flo")

    # out_v_npy = os.path.join(deepflow2_path, "output_v.npy")
    # out_h_npy = os.path.join(deepflow2_path, "output_h.npy")
    # out_v_img_path = os.path.join(deepflow2_path, "output_v.png")
    # out_h_img_path = os.path.join(deepflow2_path, "output_h.png")


    # out_v_img = fz.convert_from_file(out_v)
    # out_h_img = fz.convert_from_file(out_h)


    # Image.fromarray(out_h_img).save(out_h_img_path)
    # Image.fromarray(out_v_img).save(out_v_img_path)
    # run_deepflow2(img,img_h,out_h)
    # run_deepflow2(img,img_v,out_v)

    # out_v_array = fz.read_flow(out_v)
    # out_h_array = fz.read_flow(out_h)

    # np.save(out_v_npy,out_v_array)
    # np.save(out_h_npy,out_h_array)

    


    
    
    
    