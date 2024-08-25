
# To run the cloned file : 

## 1. Create the Virtual Environment:

``` python -m venv venv ```
## 2. Activate the Virtual Environment:

``` .\venv\Scripts\activate ```

## 3. Install Dependencies:

```pip install -r requirements.txt```


# Things i did:
## 1. Create the Virtual Environment:

``` python -m venv venv ```
## 2. Activate the Virtual Environment:

``` .\venv\Scripts\activate ```

## 3. Install Required Libraries:

``` 
    pip install opencv-python
    pip install tensorflow
    pip install scikit-learn
    pip install flask 
    pip install numpy
    pip install pandas 
```

## 4. Folder Structure
#### Explanation of folders:

* ` data/ `: Store your dataset and any processed data here.
* `models/`: Save your trained models and checkpoints.
* `src/`: Source code for your project (e.g., preprocessing, training scripts).
* `app/`: Flask application files for the web interface.
* `tests/`: Unit tests and other testing scripts.
* `.gitignore`: Specify files and directories to be ignored by Git (e.g., venv/, *.pyc).
* `requirements.txt`: List of dependencies.
* `README.md`: Project documentation.

## 5. Create a requirements.txt File:
Save your project dependencies to requirements.txt:

```pip freeze > requirements.txt```
