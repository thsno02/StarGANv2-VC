conda create -n vc python=3.9
conda.bat activate vc
pip install -r requirements.txt -i  https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113