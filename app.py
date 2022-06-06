import pickle
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)

ipe = pickle.load(open("pipe.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=="POST":
        form = request.form
        company = form.get('company')
        type = form.get('type')
        size = form.get('size')
        touch = form.get('touch')
        ips = form.get('ips')
        cpu_vender = form.get('cpu_vender')
        cpu_type = form.get('cpu_type')
        ram = form.get('ram')
        storage = form.get('storage')
        storage_type = form.get('storage_type')
        gpu_vender = form.get('gpu_vender')
        gpu_type = form.get('gpu_type')
        weight = form.get('weight')
        op_sys = form.get('op_sys')
        x=np.array([company,type,size,touch,ips,cpu_vender,cpu_type,ram,
                    storage,storage_type,gpu_vender,gpu_type,weight,op_sys])
        x=x.reshape(1,14)
        pred = str(int(np.exp(ipe.predict(x)[0])))
        return render_template('predict.html',pred=pred)
    return render_template('predict.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)
 