from flask import Flask, render_template, request, jsonify
import joblib  # Assuming you have a trained model saved as a .pkl file

app = Flask(__name__)

# Load your trained model (update the path as necessary)
model = joblib.load('RidgeModel.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        beds = float(request.form['beds'])
        baths = float(request.form['baths'])
        size = float(request.form['size'])
        lot_size = float(request.form['lot_size'])
        zipcode = int(request.form['zipcode'])
        
        # Prepare features for the model (ensure you preprocess them as needed)
        features = [[beds, baths, size, lot_size, zipcode]]
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        
        return jsonify({'price': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
