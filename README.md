# White Rice Detector
This repository contains a pretrained ResNet18 model for detecting white rice in a give image.

- **src/TrainAndEvaluate.ipynb** has the details of model training and evaluation
- In order to test the model use **src/server_model.py** by running 
  ```
  cd src
  streamlit run serve_model.py
  ```
# Sample Data
![Sample Data vis](https://i.imgur.com/q82jCQk.png)

# Understanding Model's predictions
![GradCam HeatMaps](https://i.imgur.com/b07oYnk.png)
## Model inference demo
![Model Inference GIF](demo.gif)