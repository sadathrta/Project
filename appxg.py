from flask import Flask, render_template, request
import pickle
import numpy as np





app = Flask(__name__)

model = pickle.load(open('xgmodel.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    id = int(request.form.get('id'))
    age = int(request.form.get('age'))
    education = int(request.form.get('education'))
    gender = request.form.get('gender')
    department = request.form.get('department')
    region = int(request.form.get('region'))
    recruitment = request.form.get('recruitment')
    training = int(request.form.get('training'))
    rating = float(request.form.get('rating'))
    s_len = int(request.form.get('s_len'))
    score = int(request.form.get('score'))
    kpi = int(request.form.get('kpi'))
    award = int(request.form.get('awards'))

    
    data = [gender, recruitment, department]
    encoded = ohe.transform(np.array(data).reshape(1, -1))


    scaled_score = scale.transform(np.array([score]).reshape(1, -1))

    feature = np.concatenate([np.array([ region,  education,training,age, rating, s_len, kpi, award]),
    encoded.flatten(),scaled_score.flatten()])
    
    

       
      

    # Make prediction
    output = int(model.predict([feature]))

    if output == 1:
        result = "PROMOTED"
    else:
        result = "NOT_PROMOTED"

    return render_template('result.html', prediction_text="Employee ID {} should be {}".format(id, result))

if __name__ == '__main__':
    
    app.run(port=9000,debug=True)

