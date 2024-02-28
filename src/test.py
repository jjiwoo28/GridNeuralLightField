import cv2
import numpy as np
import torch

def make_lut_u():
    return np.array([[[i, 255-i, 0] for i in range(256)]], dtype=np.uint8)

def make_lut_v():
    return np.array([[[0, 255-i, i] for i in range(256)]], dtype=np.uint8)

def save_color_mapped_channels(img_path):
    # 이미지 로드 및 YUV로 변환
    img = cv2.imread(img_path)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    # 색상 조회 테이블 생성
    lut_u, lut_v = make_lut_u(), make_lut_v()

    # U, V 채널에 LUT 적용 및 BGR로 변환
    u_mapped = cv2.LUT(cv2.cvtColor(u, cv2.COLOR_GRAY2BGR), lut_u)
    v_mapped = cv2.LUT(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR), lut_v)

    # 결과 이미지 저장
    cv2.imwrite('shed_y_channel.png', cv2.cvtColor(y, cv2.COLOR_GRAY2BGR))  # Y 채널은 단순히 그레이스케일 이미지로 저장
    cv2.imwrite('shed_u_mapped.png', u_mapped)  # 색상화된 U 채널 저장
    cv2.imwrite('shed_v_mapped.png', v_mapped)  # 색상화된 V 채널 저장
if __name__ =="__main__":
    # 이미지 로드 및 RGB에서 YUV로 변환
    img_path = 'dataset/knight_new/images/out_00_00_-381.909271_1103.376221.png'  # 이미지 경로 설정
    img = cv2.imread(img_path)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # YUV 채널 저장
    base_filename = '240206_test'  # 저장할 파일명 기본값 설정
    save_color_mapped_channels(img_path)


