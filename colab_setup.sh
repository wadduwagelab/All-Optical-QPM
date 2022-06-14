# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c nvidia
# conda install -c conda-forge matplotlib
# conda install -c conda-forge wandb

## Remove incompatible version from colab and reinstall correct version
pip -q uninstall imgaug -y
pip -q install imgaug==0.2.6

pip install -q -r /content/All-Optical-QPM/requirements.txt

echo "Downloading.."
gdown https://drive.google.com/uc?id=1ickDfs6bA-YM7RQSaMPRqFnC7YApjW8e
echo "HeLa dataset downloaded \xE2\x9C\x94"
gdown https://drive.google.com/uc?id=1CRaPVYVUs-vJA6SoXqeNhBRiqMwITbv6
echo "Bacteria dataset downloaded \xE2\x9C\x94"

# Download pretrained models
gdown https://drive.google.com/uc?id=1tHBWjNJPRHf1VX0XJicLIKs8av43_TPz
echo "Pretrained models downloaded \xE2\x9C\x94"

mkdir /content/datasets
unzip -qq hela.zip -d /content/datasets/
unzip -qq models.zip -d /content/models/
