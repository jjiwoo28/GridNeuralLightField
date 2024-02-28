import re
import numpy as np
import matplotlib.pyplot as plt


def parse_psnr_from_log(log_file_path):
    # Regular expression to find lines with PSNR values and the preceding epoch
    pattern = r"epoch (\d+)/\d+.*\n.*psnr\s+mean\s+(\d+\.\d+)"

    # Read log file
    with open(log_file_path, 'r') as file:
        log_data = file.read()

    # Find all matches for the pattern
    matches = re.findall(pattern, log_data)

    # Convert matches to numpy array
    psnr_data = np.array(matches, dtype=float)

    return psnr_data

def plot_psnr(rgbhv, rgb):
    # 전체 데이터 시리즈를 한 번에 플로팅
    plt.plot(rgbhv[:, 0], rgbhv[:, 1], label="Loss = rgb,hv")
    plt.plot(rgb[:, 0], rgb[:, 1], label="Loss = rgb")

    plt.xlabel('Epoch')
    plt.ylabel('PSNR Mean')
    plt.title("D4W128")
    plt.grid(True)
    plt.legend()

    # 파일 이름에 확장자 추가 권장 (예: .png)
    plt.savefig("240116_test.png")


    


if __name__ == "__main__":
    rgb_path = "result/Exp_rgb_240114_knight_loss_depth_d4w128_epoch1000/15-01-2024-05-31.log"
    rgbhv_path = "result/Exp_rgbhv_240114_knight_loss_depth_d4w128_epoch1000/14-01-2024-07-57.log"
    rgb = parse_psnr_from_log(rgb_path)
    rgbhv = parse_psnr_from_log(rgbhv_path)
    plot_psnr(rgbhv, rgb)
    print("test")