# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c nvidia
# conda install -c conda-forge matplotlib
# conda install -c conda-forge wandb

## Remove incompatible version from colab and reinstall correct version
pip -q uninstall imgaug -y
pip -q install imgaug==0.2.6

pip install -q -r /content/All-Optical-QPM/requirements.txt

# Download Datasets
gdown https://drive.google.com/uc?id=16vYiOBYJ8BDCBVHoOv6eqcXaLQhiS8Ik
echo "HeLa dataset downloaded ✔"
gdown https://drive.google.com/uc?id=12AdUSF7DawnqVJMzfPzqD7mUa7kmZT2L
echo "Bacteria dataset downloaded ✔"

# Download pretrained models
gdown https://drive.google.com/uc?id=1tHBWjNJPRHf1VX0XJicLIKs8av43_TPz
echo "Pretrained models downloaded ✔"

mkdir /content/datasets
unzip -qq hela.zip -d /content/datasets/

unzip -qq bacteria.zip -d /content/datasets/

unzip -qq models.zip -d /content/models/
