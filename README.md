# vision
Open Source repo demoing vision enablement in any digital surface 

## Installation
#### requires
- python >= 3.9
### PIP
- pip install -r requirements.txt
- Refresh if needed then run app
- #### OR
- pip install virtualenv
- virtualenv venv
- ./venv/Scripts/activate
- pip install -r requirements.txt

## Running APP
uvicorn src.asgi:vision --host=127.0.0.1 --port=8080 --reload


### WEB PAGES
- http://127.0.0.1:8080/vision/
- http://127.0.0.1:8080/docs