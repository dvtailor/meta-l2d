# Instructions to prepare HAM10000 dataset

Follow instructions to setup Kaggle's CLI tool (python): https://www.kaggle.com/docs/api

Run the following from root of the repository:
```
mkdir data/HAM10000
cd data/HAM10000

kaggle datasets download kmader/skin-cancer-mnist-ham10000  -p content/derm/

unzip /content/derm/skin-cancer-mnist-ham10000.zip

mkdir HAM10000_images
mv ./HAM10000_images_part_1/* HAM10000_images/
mv ./HAM10000_images_part_2/* HAM10000_images/

rm -rf content/ ham10000_images_part_1/ ham10000_images_part_2/ HAM10000_images_part_1/ HAM10000_images_part_2/ hmnist_8_8_L.csv hmnist_8_8_RGB.csv hmnist_28_28_L.csv hmnist_28_28_RGB.csv
```

Run `python preprocess.py` to extract the following files which are needed for the main script:    
* `train_data.pt`
* `validation_data.pt`
* `test_data.pt`
