import time

class Timer:
    start_time = 0
    elapsed_time = 0  # 초 단위로 측정된 시간

    @staticmethod
    def start():
        Timer.start_time = time.time()

    @staticmethod
    def stop():
        Timer.elapsed_time = time.time() - Timer.start_time
        elapsed_time_ms = Timer.elapsed_time * 1000  # 밀리초로 변환
        #print(f"time : {elapsed_time_ms} ms")
        Timer.start_time = time.time()
        
        return  elapsed_time_ms  # 초와 밀리초 단위로 반환