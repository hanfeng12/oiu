# USYD-04-HalideVision

## Dependencies
* C++ 17
* Python 3.10
* Halide 16.0.0
* PyTorch 2.0.1
* Torchvision 0.15.2
* Pytest 7.4.0
* PyBind11

## Instructions

Must have your Halide files as such:
.

├── lib

├── Images

├── Halide  

│  ---- ├── bin 

│  ---- ├── include

│  ---- ├── lib    

│  ---- └── share    

├── main.py

├── main.cpp

└── ...

To build:

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py build_ext -i
```

To run:

```shell
python main.py
```

To test:

```shell
pytest test_Halide.py
```

To benchmark:

```shell
python benchmark.py
```

