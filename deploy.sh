conda create -n vc python=3.9
conda activate vc
# conda list
# python --version
pip install -r requirements.txt -i  https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com
# conda list
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113