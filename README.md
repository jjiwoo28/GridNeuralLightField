
# GridNeuralLightField
###전처리

###학습

    python src/train_neulf_grid.py --data_dir dataset/knight --factor 1 --exp_name exp_name --grid_n 2
padding 모드
    python src/train_grid_shared.py --data_dir dataset/01_1to6 --factor 1 --exp_name bmw_01_1to6_padding8 --grid_n 32 --padding 8 


* grid_n 2라면 2 by 2 grid를 생성합니다.
* padding 값은 각 grid의 상,하,좌,우에 입력한 픽셀만큼의 데이터를 추가 제공하여 학습을 진행힙니다.

###랜더링
    python src/train_neulf_grid.py --data_dir dataset/knight --factor 1 --exp_name exp_name --grid_n 2 --render_only
* --render_only 를 추가하면 학습을 진행하지 않고 video를 생성합니다.
