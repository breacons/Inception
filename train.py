'''
Copyright
Jelen forráskód a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott
"Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült.
A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning
A forráskódot GPLv3 licensz védi. Újrafelhasználás esetén lehetőség szerint kérejük
az alábbi szerzőt értesíteni.
Az Inception V3 modell ebben a cikkben kerül bemutatásra:
https://arxiv.org/abs/1512.00567
A kód elkészítéséhez az alábbi források kerültek felhasználásra:
https://keras.io/applications/
https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
Adatokat innen lehet regisztráció után letölteni:
https://www.kaggle.com/c/dogs-vs-cats
Az adatokat az alábbiak szerint kell könyvtárba rendezni:
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
2016 (c) Tóth Bálint Pál (toth.b kukac tmit pont bme pont hu)
'''
from PIL import Image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np

# full matrixes
np.set_printoptions(threshold=np.inf, precision=4)

# függvény az eredmények kiírására
def print_result(result):
    cf_matrix = np.zeros((10, 10), dtype=int)
    cls = list(test_generator.class_indices.values())  # osztályok azonosítója

    for idx, item in enumerate(result):
        predicted_class = cls[np.argmax(item)]
        true_class = test_generator.classes[idx]

        cf_matrix[true_class][predicted_class] += 1
    print("Címkék:")
    print(list(test_generator.class_indices.keys()))

    print("Konfúziós mátrix:")
    print(cf_matrix)

    true_positive = np.zeros((10), dtype=int)
    precision = np.zeros((10), dtype=float)
    recall = np.zeros((10), dtype=float)
    f1 = np.zeros((10), dtype=float)

    for i in range(10):
        true_positive[i] += cf_matrix[i][i]

    precision_s = np.sum(cf_matrix, axis=0)
    recall_s = np.sum(cf_matrix, axis=1)

    for i in range(10):
        if precision_s[i] != 0:
            precision[i] = true_positive[i] / precision_s[i]
        if recall_s[i] != 0:
            recall[i] = true_positive[i] / recall_s[i]
        if (precision[i] + recall[i]) > 0:
            f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

    accuracy = np.sum(true_positive) / nb_test_samples

    print("True positive: ", true_positive, " average: ", np.sum(true_positive) / nb_test_samples)
    print("Precision: ", precision, " average: ", np.sum(precision) / nb_test_samples)
    print("Recall: ", recall, " average: ", np.sum(recall) / nb_test_samples)
    print("Accuracy: ", accuracy)
    print("F1: ", f1, " average: ", np.sum(f1) / nb_test_samples, "\n\n")


# a bemenő képek mérete (Inception V3 bemenete 299x299)
img_height = 299
img_width = 299

# a tanító és validációs adatbázis elérési útvonala, teszt adatbázis most nincs
train_data_dir = 'data/train'
test_data_dir = 'data/test'
validation_data_dir = 'data/validation'
# a tanító és validációs adatok száma
nb_train_samples = 1110
nb_test_samples = 16
nb_validation_samples = 150
# epoch szám
nb_epoch = 1

# előtanított modell betöltése, a fully-connected rétegek nélkül
base_model = InceptionV3(weights='imagenet', include_top=False)
# az utolsó konvolúciós réteg utána egy global average pooling réteget teszünk, ez rögtön "lapítja" (flatten) a 2D konvolúciót
x = base_model.output
x = GlobalAveragePooling2D()(x)
# ezután hozzáadunk egy előrecsatolt réteget ReLU aktivációs függvénnyel
x = Dense(1024, activation='relu')(x)
# 10 kimenet a 10 osztályhoz
predictions = Dense(10, activation='sigmoid')(x)
# a model létrehozása
model = Model(input=base_model.input, output=predictions)

# két lépésben fogjuk tanítani a hálót
# az első lépésben csak az előrecsatolt rétegeket tanítjuk, a konvolúciós rétegeket befagyasztjuk
for layer in base_model.layers:
    layer.trainable = False
# lefordítjuk a modelt (fontos, hogy ezt a rétegek befagyasztása után csináljuk"
# categorical_crossentropy mivel több osztály van
model.compile(optimizer='adadelta', loss='categorical_crossentropy')

# kép felkészítése a betöltésre és adatdúsításra
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=32,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_height, img_width), batch_size=32,
                                                  class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width),
                                                              batch_size=32, class_mode='categorical')


print("Tanítás előtt:")
before_learning = model.predict_generator(test_generator, val_samples=nb_test_samples)
print(model.evaluate_generator(test_generator, val_samples=nb_test_samples))
print_result(before_learning)

# ez a függvény egyszerre végzi az adatdúsítást és a háló tanítását
print("Első lépés:")
model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
                    validation_data=validation_generator, nb_val_samples=nb_validation_samples)
first_stage = model.predict_generator(test_generator, val_samples=nb_test_samples)

print(model.evaluate_generator(test_generator, val_samples=nb_test_samples))
print_result(first_stage)

# most már van egy célra betanított osztályozónk, ami az Inception V3 előtanított hálót követi
# most jön a második lépés, aminek a során a konvolúciós háló mélyebb rétegeit fagyasztjuk
# felsőbb rétegeit pedig tovább tanítjuk

# ehhez először nézzük meg a háló felépítését
# print("Az Inception V3 konvolúciós rétegei:")
# for i, layer in enumerate(base_model.layers):
# print(i, layer.name)

# majd a hálónak csak az első 172 rétegét fagyasztjuk, a többit pedig engedjük tanulni
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# ez után újra le kell fordítanunk a hálót, hogy most már az Inception V3 felsőbb rétegei tanuljanak
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay=0.1), loss='binary_crossentropy')

# és ismét indítunk egy tanítást, ezúttal nem csak az előrecsatolt rétegek,
# hanem az Inception V3 felső rétegei is tovább tanulnak


model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
                    validation_data=validation_generator, nb_val_samples=nb_validation_samples)

last_stage = model.predict_generator(test_generator, val_samples=nb_test_samples)

print("Tanítás vége.")

model.save('inception.h5')

nb_prediction = 16
true_prediction = 0

print("Végső tanítás:")
print(model.evaluate_generator(test_generator, val_samples=nb_test_samples))
print_result(last_stage)
