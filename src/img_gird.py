import cv2
import os
import numpy as np

def split_into_grids(image, n):
    """
    이미지를 n x n 그리드로 분할합니다.

    :param image: (H, W, 3) 형태의 NumPy 배열로 된 이미지
    :param n: 그리드의 수 (세로 및 가로)
    :return: 분할된 그리드의 리스트
    """
    H, W, _ = image.shape
    grid_height = H // n
    grid_width = W // n

    grids = []
    for i in range(n):
        for j in range(n):
            grid = image[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width, :]
            grids.append(grid)
    
    
    return np.array(grids)





def test1():
    
    # 이미지 경로 지정
    image_path = 'dataset/knight/images/out_00_00_-381.909271_1103.376221.png'

    # 이미지 읽기
    image = cv2.imread(image_path)
    #image = np.array(image)
    # 그리드로 분할 (예: 3x3 그리드)
    n = 2
    grids = split_into_grids(image, 8)

    # 첫 번째 그리드 이미지 저장
   

    folder_path = "GridTest8"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i ,grid in enumerate(grids):
        path = os.path.join(folder_path, f"{i:03}_gird.jpg")
        cv2.imwrite(path, grid)


if __name__ == "__main__":
    test1()