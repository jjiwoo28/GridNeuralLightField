from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import re
import os
def visualize_epochs_from_json_file(input_filename, output_filename, data_type):
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
    epoch_data = np.zeros((grid_n, grid_n), dtype=float)  # dtype을 float으로 변경
    max_value = 0
    for row in json_data:
        for record in row:
            i = record["index_i"]
            j = record["index_j"]
            value = record[data_type]
            if value is None:
                raise ValueError(f"JSON 데이터에 None 값이 포함되어 있습니다. 위치: ({i}, {j})")

            epoch_data[i, j] = value
            max_value = max(max_value, value)

    # 이미지 초기화
    img_size = 1024
    grid_size = img_size // grid_n
    img = Image.new('RGB', (img_size, img_size), color='white')
    draw = ImageDraw.Draw(img)

    # 폰트 크기를 그리드 크기에 적응적으로 조정
    font_size =  int(grid_size / 4)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # 그리드 별로 색상 및 텍스트 렌더링
    for i in range(grid_n):
        for j in range(grid_n):
            value = epoch_data[i, j]
            color_value = int((value / max_value) * 255)
            color = (color_value, 255 - color_value, 100)
            draw.rectangle([j*grid_size, i*grid_size, (j+1)*grid_size, (i+1)*grid_size], fill=color)
            if data_type == "epoch":
                text = str(int(value))
            else:
                text = "{:.2f}".format(value)  # 소수점 둘째 자리까지 표현
            textwidth, textheight = draw.textsize(text, font=font)
            x = j*grid_size + (grid_size - textwidth) / 2
            y = i*grid_size + (grid_size - textheight) / 2
            draw.text((x, y), text, fill='black', font=font)

    # 이미지 저장
    img.save(output_filename)

def blend_images(image_path1, image_path2, output_path, weight=50):
    """
    두 이미지를 받아 1024x1024 크기로 조정한 후 주어진 비율로 블렌딩합니다.

    :param image_path1: 첫 번째 이미지 파일 경로
    :param image_path2: 두 번째 이미지 파일 경로
    :param output_path: 블렌드된 이미지를 저장할 파일 경로
    :param weight: 첫 번째 이미지의 블렌딩 비율 (0-100)
    """
    # 이미지 로드
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # 이미지 크기 확인 및 조정
    if image1.size != (1024, 1024):
        image1 = image1.resize((1024, 1024))
    if image2.size != (1024, 1024):
        image2 = image2.resize((1024, 1024))

    # 블렌딩 비율 계산
    alpha = weight / 100.0

    # 이미지 블렌드
    blended_image = Image.blend(image1, image2, alpha)

    # 결과 이미지 저장
    blended_image.save(output_path)


def generate(filename):

    out_path = os.path.join("result_json_img", filename)

    in_path = os.path.join("result_json",f"{filename}.json")

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    



    visualize_epochs_from_json_file(in_path, os.path.join(out_path,f'{filename}_epoch.png'), "epoch")
    visualize_epochs_from_json_file(in_path, os.path.join(out_path,f'{filename}_psnr.png'), "psnr")
    visualize_epochs_from_json_file(in_path, os.path.join(out_path,f'{filename}_loss.png'), "loss")
    weight = 70
    
    blend_images('merge_imgs_gt/merged_image_0.png', os.path.join(out_path,f'{filename}_epoch.png') ,os.path.join(out_path,f'{filename}_epoch_blend.png') , weight )
    blend_images('merge_imgs_gt/merged_image_0.png', os.path.join(out_path,f'{filename}_psnr.png') , os.path.join(out_path,f'{filename}_psnr_blend.png'),weight)
    blend_images('merge_imgs_gt/merged_image_0.png', os.path.join(out_path,f'{filename}_loss.png') , os.path.join(out_path,f'{filename}_loss_blend.png'),weight)
    



# 사용 예시
if __name__ == "__main__":

    generate("2_test_lossclamp_knight16_lr_d4w32")
    generate("2_test_lossclamp_knight32_lr_d4w16")
    generate("2_test_lossclamp_knight32_lr_d4w32")

    # visualize_epochs_from_json_file('result_json/grid_n-32,d-4,w-32.json', 'result_json/grid_n-32,d-4,w-32.json_img_epoch.png', "epoch")
    # visualize_epochs_from_json_file('result_json/grid_n-16,d-4,w-64.json', 'result_json/grid_n-16,d-4,w-64.json_img_epoch.png', "epoch")
    # visualize_epochs_from_json_file('result_json/grid_n-8,d-4,w-64.json', 'result_json/grid_n-8,d-4,w-64.json_img_epoch.png', "epoch")
    # visualize_epochs_from_json_file('result_json/grid_n-32,d-4,w-32.json', 'result_json/grid_n-32,d-4,w-32.json_img_psnr.png', "psnr")
    # visualize_epochs_from_json_file('result_json/grid_n-16,d-4,w-64.json', 'result_json/grid_n-16,d-4,w-64.json_img_psnr.png', "psnr")
    # visualize_epochs_from_json_file('result_json/grid_n-8,d-4,w-64.json', 'result_json/grid_n-8,d-4,w-64.json_img_psnr.png', "psnr")
    # visualize_epochs_from_json_file('result_json/grid_n-32,d-4,w-32.json', 'result_json/grid_n-32,d-4,w-32.json_img_loss.png', "loss")
    # visualize_epochs_from_json_file('result_json/grid_n-16,d-4,w-64.json', 'result_json/grid_n-16,d-4,w-64.json_img_loss.png', "loss")
    # visualize_epochs_from_json_file('result_json/grid_n-8,d-4,w-64.json', 'result_json/grid_n-8,d-4,w-64.json_img_loss.png', "loss")
    # weight = 70
    # visualize_epochs_from_json_file('result_json/2_test_lossclamp_knight32_lr_d4w64.json', 'result_json/2_test_lossclamp_knight32_lr_d4w64_epoch.png', "epoch")
    # visualize_epochs_from_json_file('result_json/2_test_lossclamp_knight32_lr_d4w64.json', 'result_json/2_test_lossclamp_knight32_lr_d4w64_psnr.png', "psnr")
    # visualize_epochs_from_json_file('result_json/2_test_lossclamp_knight32_lr_d4w64.json', 'result_json/2_test_lossclamp_knight32_lr_d4w64_loss.png', "loss")
    # weight = 70
    
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/2_test_lossclamp_knight32_lr_d4w64_epoch.png' ,'result_json/2_test_lossclamp_knight32_lr_d4w64_epoch_blend.png' , weight )
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/2_test_lossclamp_knight32_lr_d4w64_psnr.png' , 'result_json/2_test_lossclamp_knight32_lr_d4w64_psnr_blend.png',weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/2_test_lossclamp_knight32_lr_d4w64_loss.png' , 'result_json/2_test_lossclamp_knight32_lr_d4w64_loss_blend.png',weight)
    

    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-32,d-4,w-32.json_img_epoch.png' ,'result_json/grid_n-32,d-4,w-32.json_img_epoch_blend.png' , weight )
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-16,d-4,w-64.json_img_epoch.png', 'result_json/grid_n-16,d-4,w-64.json_img_epoch_blend.png', weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-8,d-4,w-64.json_img_epoch.png' , 'result_json/grid_n-8,d-4,w-64.json_img_epoch_blend.png' ,weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-32,d-4,w-32.json_img_psnr.png' , 'result_json/grid_n-32,d-4,w-32.json_img_psnr_blend.png',weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-16,d-4,w-64.json_img_psnr.png' , 'result_json/grid_n-16,d-4,w-64.json_img_psnr_blend.png',weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-8,d-4,w-64.json_img_psnr.png' , 'result_json/grid_n-8,d-4,w-64.json_img_psnr_blend.png',weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-32,d-4,w-32.json_img_loss.png' , 'result_json/grid_n-32,d-4,w-32.json_img_loss_blend.png',weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-16,d-4,w-64.json_img_loss.png' , 'result_json/grid_n-16,d-4,w-64.json_img_loss_blend.png',weight)
    # blend_images('merge_imgs_gt/merged_image_0.png', 'result_json/grid_n-8,d-4,w-64.json_img_loss.png' ,'result_json/grid_n-8,d-4,w-64.json_img_loss_blend.png',weight)
