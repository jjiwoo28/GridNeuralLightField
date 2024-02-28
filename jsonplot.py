import json
import numpy as np
import matplotlib.pyplot as plt
import re

def visualize_epochs_from_json_file(input_filename,  output_filename):
    """
    JSON 파일을 읽어서 epoch 값을 시각화하고, 이미지 파일로 저장합니다.
    파일 이름에서 grid_n, d, w 값을 파싱하여 시각화에 사용합니다.

    :param input_json_filename: JSON 데이터가 저장된 파일 이름
    :param input_filename: 입력 파일 이름 (예: 'grid_n-8,d-4,w-64')
    :param output_filename: 시각화 이미지를 저장할 파일 이름
    """
    # JSON 파일 읽기
    with open(input_filename, 'r') as file:
        json_data = json.load(file)

    # 파일 이름 파싱
    
    #pattern = r'grid_n-(\d+),d-(\d+),w-(\d+)'
    pattern = r'knight(\d+)_lr_d(\d+)w(\d+)'
    match = re.search(pattern, input_filename)
    if not match:
        raise ValueError("파일 이름이 올바른 형식이 아닙니다.")
    
    grid_n = int(match.group(1))
    d = int(match.group(2))
    w = int(match.group(3))

    # JSON 데이터를 2차원 배열로 변환
    epoch_data = np.zeros((grid_n, grid_n))
    for row in json_data:
        for record in row:
            i = record["index_i"]
            j = record["index_j"]
            epoch = record["epoch"]
            epoch_data[i, j] = epoch

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(epoch_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Epoch')
    plt.title(f'Epoch Visualization (Grid: {grid_n}, d: {d}, w: {w})')
    plt.xlabel('Index j')
    plt.ylabel('Index i')
    
    # 이미지 파일로 저장
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    visualize_epochs_from_json_file('result_json/2_test_lossclamp_knight32_lr_d4w64.json',"result_json/2_test_lossclamp_knight32_lr_d4w64.json.png" )
# 예시 사용
# visualize_epochs_from_json_file('path_to_json_file.json', 'grid_n-8,d-4,w-64', 'output_image.png')

