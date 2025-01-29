# Face Tracking Age and Gender Predictions

This project performs real-time face tracking and predicts age and gender using OpenCV and a pre-trained deep learning model.

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/yashdha1/Face-tracking-age-and-gender-predictions.git
cd Face-tracking-age-and-gender-predictions
```

### **2. Download Required Model Files**
The following model files are required for age and gender prediction:

- `age_net.caffemodel`
- `deploy_age.protext`
- `deploy_gender.protext`
- `gender_net.caffemodel`

#### **Download the files from:**
[GitHub - age-and-gender-classification](https://github.com/eveningglow/age-and-gender-classification/blob/master/model/age_net.caffemodel)

#### **Move them into the `modelWeights` folder:**
```bash
mkdir modelWeights
mv path/to/downloaded/files/* modelWeights/
```

### **3. Install Dependencies**
Make sure you have Python installed (preferably Python 3.8+). Then, install the required dependencies:
```bash
pip install opencv-python numpy facenet-pytorch torch torchvision
```

### **4. Run the Application**
Once everything is set up, run the script to start face tracking and predictions:
```bash
python face_recog.py
```

### **5. Quit the Application**
To stop the program, press **'q'** in the OpenCV window.

---
## **License** 

feel free to clone and make changes

