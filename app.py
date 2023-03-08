import pickle
from flask import Flask, request, render_template

# Load the GBM model
with open('gbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app instance
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('predict.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the distance from the form
    distance = float(request.form['distance'])

    # Use the model to make a prediction
    fare = model.predict([[distance]])[0]

    # Render the prediction result in a new page
    return render_template('result.html', fare=fare)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
