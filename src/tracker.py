import json

class ResultTracker:
    def __init__(self, grid_size, filename="grid_data.json"):
        self.grid_size = grid_size
        self.filename = filename
        self.data = [[{"index_i": i, "index_j": j, "epoch": None, "psnr": None, "loss": None} 
                      for j in range(grid_size)] for i in range(grid_size)]

    def add_measurement(self, index_i, index_j, epoch, psnr=None, loss=None):
        epoch = int(epoch)  # Epoch 값을 정수로 변환
        measurement = {"index_i": index_i, "index_j": index_j, "epoch": epoch, "psnr": psnr, "loss": loss}
        self.data[index_i][index_j] = measurement

    def save(self):
        with open(self.filename, 'w') as file:
            json.dump(self.data, file, indent=4)
    def save(self ,file_name):
        with open(file_name, 'w') as file:
            json.dump(self.data, file, indent=4)        

    def load(self):
        try:
            with open(self.filename, 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print(f"파일 '{self.filename}'을(를) 찾을 수 없습니다. 새 데이터를 사용합니다.")
            self.data = [[{"index_i": i, "index_j": j, "epoch": None, "psnr": None, "loss": None} 
                          for j in range(self.grid_size)] for i in range(self.grid_size)]

    def get_data(self):
        return self.data

    def get_measurement(self, index_i, index_j):
        return self.data[index_i][index_j]

    def update_measurement(self, index_i, index_j, epoch=None, psnr=None, loss=None):
        if epoch is not None:
            self.data[index_i][index_j]["epoch"] = int(epoch)
        if psnr is not None:
            self.data[index_i][index_j]["psnr"] = psnr
        if loss is not None:
            self.data[index_i][index_j]["loss"] = loss
