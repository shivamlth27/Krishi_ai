from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
