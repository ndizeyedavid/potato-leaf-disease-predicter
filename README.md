## Another python project (_What's wrong with me!_)

#### A'ight let's get this over with again, the steps to make it run are simple but first huge thanks to **Rizwan Saeed** on **Kaggle** for providing the dataset used to train the model

ok, now let's start:

#### sample
![Screenshot 2024-12-22 124053](https://github.com/user-attachments/assets/3f1d5740-4055-4fe5-ab47-b59fd0046bf9)


1. clone the project
```bash
git clone https://github.com/ndizeyedavid/potato-leaf-disease-predicter.git
```

2. enter the project directory
```bash
cd potato-leaf-disease-predicter
```

3. install all libraries and packages
```bash
pip install -r requirements.txt
```

4. Create a virtual environment
```bash
python -m venv venv
```

5. start the virtual environment
   Windows
```bash
.\venv\Scripts\activate
```

  Mac & Linux
```bash
source .\venv\Scripts\activate
```

6. for real time prediction with your device's camera
```bash
py realtime_prediction.py
```

7. for static prediction with images in a directory
   > in the ```App.py``` change the ```test_image``` with your desired source image then run ```py app.py```

8. If you want to the train with your own data, use the ```model_trainer.py```

### 💫 Let me know what you think of this project or mi skills 😊
