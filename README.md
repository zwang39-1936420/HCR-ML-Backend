This is the backend model repo. To access the HCR website, please access this URL: (https://handwriting-recognition-app-2bdcf9aba5c3.herokuapp.com/)

# *Offline Handwritten Character Recognition*

### The purpose of this notebook is to build up an simple machine learning model for offline handwritten text recognition by using segmentation and classification. 

I borrowed initial source code here, and made further adjustment based on that. Initial-code: https://www.kaggle.com/code/aman10kr/offline-handwritten-text-ocr



what changes I made
* **1** : I ignored the number characters in the original dataset to only focus on alphabetic character recognition.
* **2** : I fine-tuned the CNN model: 
    * *Batch Normalization* : Add Batch Normalization layers to normalize the inputs between layers, which can accelerate training and improve model convergence.

    * *Use Different Activation Functions*: Try different activation functions for hidden layers, such as Leaky ReLU or Parametric ReLU, to introduce non-linearity.

    All in all, the changes improved model's performance on both training set and validation set. 
* **3** : Fixed few bugs in the original code.  

what new code I implemented.
* backend:  I created the model_batch to improve the overall performance (This repo)
* frontend: I coded full-stack structure the HCR recognizing web-app. I used React for the frontend and Flask for the backend development. (Repo URL: https://github.com/zwang39-1936420/HCR-ML-Frontend)
