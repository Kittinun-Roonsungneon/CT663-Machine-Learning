from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = FastAPI()

class Item(BaseModel):
    Age: int
    EstimatedSalary: int

# Load the pre-trained model
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

# Example feature scaling function
def scale_features(age, salary):
    scaler = StandardScaler()
    scaled_age = scaler.fit_transform(np.array(age).reshape(-1, 1))
    scaled_salary = scaler.transform(np.array(salary).reshape(-1, 1))
    return scaled_age, scaled_salary

# Example prediction function
def predict(classifier, age, salary):
    prediction = classifier.predict(np.array([age, salary]).reshape(1, -1))
    return prediction[0]

@app.post("/predict/")
async def predict_item(item: Item):
    scaled_age, scaled_salary = scale_features(item.Age, item.EstimatedSalary)
    prediction = predict(classifier, scaled_age, scaled_salary)
    return {"prediction": prediction}