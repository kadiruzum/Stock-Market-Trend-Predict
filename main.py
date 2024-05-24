import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CSV dosyasını oku
data = pd.read_csv('/kaggle/input/datasetmultiple/dataset.csv')

hisse_senedi = 'KCHOL.IS'

# Veri setinden belirli hisse senedi sembolüne sahip satırları seçme
df = data[data['Stock'] == hisse_senedi]

# Gerekli işlemleri yapın (örneğin, sadece tarih ve kapanış fiyatını içeren bir DataFrame oluşturun)
df = pd.DataFrame(df, columns=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'])

print(df)

# Matplotlib için grafik oluştur
fig, ax = plt.subplots()

# Grafikte kapanış fiyatlarını ve tarihleri çiz
ax.plot(df['Date'], df['Close'])

# Eksenleri formatla
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title('Close Stock Price History')
plt.xlabel('Date')
plt.ylabel('Closing Stock Price in $')
plt.show()

train, test = train_test_split(df, test_size=0.20, random_state=0)


# Eğitim verisi için model oluştur
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']
model = LinearRegression()
model.fit(X_train, y_train)

# Grafik için tarih etiketlerini oluşturma
dates = pd.to_datetime(train['Date'])  # Tarih sütununu datetime nesnesine dönüştürme

# Train set grafiği
plt.title('Linear Regression | Price vs Time')
plt.scatter(dates, y_train, edgecolor='w', label='Actual Price')
plt.plot(dates, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')

# X ekseninde tarih formatını özelleştirme
date_form = DateFormatter("%Y-%m-%d")  # Tarih formatı
plt.gca().xaxis.set_major_formatter(date_form)  # X ekseninde tarih formatını ayarlama
plt.gcf().autofmt_xdate()  # Tarih etiketlerini otomatik olarak döndürme

plt.legend()
plt.show()