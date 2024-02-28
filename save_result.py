from src.asem_gird import make_result_with_json2
from json2img import generate
import os
import glob

def get_json_filenames(folder_path):
    # 폴더 경로 내의 모든 .json 파일을 찾음
    search_path = os.path.join(folder_path, '*.json')
    file_paths = glob.glob(search_path)

    # 파일 경로에서 파일명만 추출
    file_names = [os.path.basename(path) for path in file_paths]

    return file_names

def save_result(json_name):
    try:
        generate(json_name)
        make_result_with_json2(json_name)
    except ValueError as e:
        print(e)
        print(f"Excpton : {json_name}")



if __name__ =="__main__":
    save_result("1_240124_test_lossclamp_knight16_lr_d4w32")
    # paths = get_json_filenames("result_json")
    # for p in paths:
    #     save_result(p.replace('.json',''))



