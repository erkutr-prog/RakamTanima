#Rakam tanıma 
#Kütüphanelerin kurulumu

import tensorflow as tf 
import matplotlib.pyplot as plt

#MNIST datasetinin yüklenmesi
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Parametrelerin belirlenmesi
batch_size = 128 
classes = 10
epochs = 5

#Görsel formatının ayarlanması
img_rows, img_cols = 28,28

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#Integer olan verilerin binary matrise çevirilmesi
y_train= tf.keras.utils.to_categorical(y_train, classes)
y_test = tf.keras.utils.to_categorical(y_test, classes)


#Model oluşturma 

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape = input_shape),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(classes, activation='softmax')
])

#Model özeti
model.summary()

#Modelin derlenmesi
model.compile(optimizer=tf.keras.optimizers.Adadelta(),loss= tf.keras.losses.categorical_crossentropy,metrics= ['accuracy'])


#Modelin eğitimi
model.fit(x_train,y_train, 
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test,y_test)                        
)

#Test için rastgele derğer seçme 
test_img = x_test[32]
y_test[32] 
plt.imshow(test_img.reshape(28,28))

#Modelin testi 
testdata = x_test[32].reshape(1,28,28,1)
pre= model.predict(testdata, batch_size=1)
print(pre)


