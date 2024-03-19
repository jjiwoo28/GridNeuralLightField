


# GridNeuralLightField
### 전처리
    python src/llffProcess.py --data_dir dataset/knight --factor 1 #4D light field
    python src/llffProcess_xyzxyz.py --data_dir dataset/knight --factor 1 #6D light field

### 학습

    python src/train_neulf_grid.py --data_dir dataset/knight --factor 1 --exp_name exp_name --grid_n 2

padding 모드

    python src/train_grid_shared.py --data_dir dataset/01_1to6 --factor 1 --exp_name bmw_01_1to6_padding8 --grid_n 32 --padding 8 



* grid_n 2라면 2 by 2 grid를 생성합니다.
* padding 값은 각 grid의 상,하,좌,우에 입력한 픽셀만큼의 데이터를 추가 제공하여 학습을 진행힙니다.
* --use_6D 를 추가하면 6D light field로 학습이 진행 됩니다.

### 랜더링

    python src/train_neulf_grid.py --data_dir dataset/knight --factor 1 --exp_name exp_name --grid_n 2 --render_only

* --render_only 를 추가하면 학습을 진행하지 않고 video를 생성합니다.
* --use_6D 를 추가하면 6D light field로 랜더링이 진행됩니다.
* --load_epoch 을 통해 특정 epoch의 모델을 불러와 랜더링할 수 있습니다.

# DynamicNeuralLightField
### 전처리
* 내부의 val_frames과 val_images를 수정한 후
    python src/llffProcess_dy.py --data_dir dataset/knight
* frame의 갯수만큼 pose와 uvst의 npy가 생성됩니다. (*용량주의*)
### 학습
    python src/train_neulf_dy.py --data_dir dataset/knight --exp_name knight
### 랜더링
* 내부의 렌더링 함수를 수정한 후
    python src/demo_rgb_dy.py --load_exps dataset/knight --exp_name knight 
