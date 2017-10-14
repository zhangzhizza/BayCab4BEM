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
