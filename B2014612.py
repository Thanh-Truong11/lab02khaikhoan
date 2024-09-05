from flask import Flask, request, render_template
import joblib
import numpy as np

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Đọc mô hình và scaler từ file
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Định nghĩa giao diện và chức năng phân lớp
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Nhận dữ liệu đầu vào từ người dùng
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Dự đoán lớp
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return f'Prediction: {prediction[0]}'
    
    return '''
        <form method="post">
            Sepal Length: <input type="text" name="sepal_length"><br>
            Sepal Width: <input type="text" name="sepal_width"><br>
            Petal Length: <input type="text" name="petal_length"><br>
            Petal Width: <input type="text" name="petal_width"><br>
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
