conda create -n vc python=3.9
source activate vc
# conda config
conda config --set ssl_verify true 
conda config --set proxy_servers.http http://127.0.0.1:7890
conda config --set show_channel_urls yes
pip install -r requirements.txt -i  https://pypi.doubanio.com/simple/  --trusted-host pypi.doubanio.com
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch