import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ file iris.csv
data = pd.read_csv('/content/sample_data/iris.csv')

# In ra tên các cột để kiểm tra
print(data.columns)

# Tiền xử lý dữ liệu
# Giả sử rằng file iris.csv đã có sẵn tiêu đề và không cần xử lý đặc biệt
# Tách dữ liệu ra thành features và labels
X = data.drop('variety', axis=1)  # features  # Thay 'species' thành 'Species' nếu đó là tên cột chính xác
y = data['variety']  # labels  # Thay 'species' thành 'Species' nếu đó là tên cột chính xác

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiền xử lý: Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Lưu mô hình và scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')