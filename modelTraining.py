# import raznih library-a
import os # za interakciju sa operacijskim sustavom / kreiranje path-a
import cv2 # za pred-procesiranje podataka i korištenje VideoCapture u play.py skripti
import imutils # za rotiranje trening slika
import random 
import numpy as np # za matrice i pretvaranje inputa u numerička podatke
import seaborn as sns # za vizualizaciju podataka
import matplotlib.pyplot as plt # za vizualizaciju podataka
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential # za kreiranje modela
from sklearn.metrics import classification_report, confusion_matrix # za prikaz performanse modela
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D # za kreiranje modela
plt.style.use('ggplot')

# definiranje array-a u koji spremamo predprocesirane slike
training_data = []
DATADIR = 'rps/' # mapa sa trening slikama
CATEGORIES = ['rock','paper','scissors','nothing']
IMG_SIZE = 80

# funkcija u kojoj prolazimo kroz sve klase iz "CATEGORIES" array-a i za svaku klasu izvršavamo predprocesiranje - 
# konverzija u sive tonove, rotiranje...
def processTrainingData():
    IMG_SIZE = 80
    for category in CATEGORIES:
        Path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(Path):
            try:
                # metoda imread() učitava sliku iz određenog patha, zatim sa cv2.IMREAD_GRAYSCALE pretvara sve 
                # slike u slike sa sivim tonovima (single channel grayscale images)
                img_array = cv2.imread(os.path.join(Path, img), cv2.IMREAD_GRAYSCALE)
                # resizeanje svih slika na veličinu (80, 80)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # rotiranje slika
                for angle in np.arange(0, 360, 60):
                    rotated = imutils.rotate(new_array, angle)
                    training_data.append([rotated, class_num])
            except Exception as e:
                pass

processTrainingData()

random.shuffle(training_data)

print("Broj slika: ", len(training_data))

x = []
y = []
for features,labels in training_data:
    x.append(features)
    y.append(labels)

X = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE ,1)
print("The Shape Of X-axis:" , X.shape)

train_size = int(X.shape[0]*0.7)
X_train , X_test = X[:train_size,:,:,:] , X[train_size:,:,:,:]
y_train , y_test = y[:train_size] ,y[train_size:]
print('The Shape of X Train:', X_train.shape)
print('The shape of X Test:', X_test.shape)

# data normalization - dobijemo raspon vrijednosti od 0 do 1
X_train = X_train/255.0
X_test = X_test/255.0

# kreiranje CNN modela
model = Sequential()
model.add(Conv2D(16,(3,3),  input_shape = (80,80,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))

# konfiguriranje modela za trening
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train,np.array(y_train),batch_size=32,epochs=8, verbose=1 , validation_split=0.3)

print("Summary of model: ", model.summary())

model.save('RockPaperScissor_60_DEGREE_GRAYSCALE-1.h5')

# testiranje modela
model = load_model('RockPaperScissor_60_DEGREE_GRAYSCALE-1.h5')
label = model.predict(X_test)
classes =np.argmax(label,axis=1)

# confusion matrix i classification report koristim za definiranje 
# performanse klasifikacijskog modela
cm = confusion_matrix(np.array(y_test), classes)
cr = classification_report(np.array(y_test), classes)
print("The Classification Report: ")
print(cr)

# model prediction heatmap
plt.figure(figsize=(16,6))
plt.title('Predicted Vs Truth')
sns.heatmap(cm, annot=True,fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Truth')
# plt.savefig('plots\Predicted_VS_Truth.png')
plt.show()

# prikazivanje grafa na kojem se vidi točnost modela
plt.figure(figsize=(12,4))
plt.plot(history.history['accuracy'],marker='o')
plt.plot(history.history['val_accuracy'],marker='o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
# plt.savefig('plots\ModelAccuracy.png')
plt.show()

# prikazivanje gubitka (loss)
plt.figure(figsize=(12,4))
plt.plot(history.history['loss'],marker='o')
plt.plot(history.history['val_loss'],marker='o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
# plt.savefig('plots\ModelLoss.png')
plt.show()