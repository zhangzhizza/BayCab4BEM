# BayCab4BEM
## To run (Linux environment)
1. Create a virtual environment. 
```shell
virtualenv --python=python3.5 virt_env
```
2. Enter the virtual environment
```shell
source path_to_virt_env/bin/activate
```
3. Install the required depedencies
```shell
pip install -U -r path_to_requirements.txt
```
4. Run the main
```shell
python src/testWithSim.py
```
or
```shell
python src/testWithData.py
```
# Known issues
1. If Pystan has compile problem, try
```shell
sudo apt-get install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
```