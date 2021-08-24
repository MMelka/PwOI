from PIL import Image  # biblioteka do obrazów
from glob import glob  # zrobienie listy plików
from os.path import sep, join, splitext  # separator,łącznik,podział nazw
from skimage.feature import greycomatrix, greycoprops # bliźniaczy pakiet do scikit-learn'a. Do wyliczania cech i macierzy zdarzeń
from pandas import DataFrame  # umożliwia manipulacje dużymi zbiorami danych m.in, tabelami. Użyty do zapisu pliku csv
from itertools import product
import numpy as np

feature_names = ('dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM')  # nazwy cech tekstur jakie mają być wyznaczone

distances = (1, 3, 5)  #
#
angles = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)  # określenie dla jakich kątów bedzie wyliczana macierz zdarzeń


def get_full_names():
    dist_str = ('1', '2', '5') # określenie dla jakich dystansów bedzie wyliczana macierz zdarzeń
    angles_str = '0deg, 45deg, 90deg, 135deg'.split(',')
    return ['_'.join(f) for f in product(feature_names, dist_str, angles_str)]


# UTOWRZENIE MACIERZY ZDARZEŃ


def get_glcm_feature_array(patch):
    patch_64 = (patch / np.max(patch) * 63).astype('uint8')
    glcm = greycomatrix(patch_64, distances, angles, 64, True, True) # utworzenie macierzy zdarzeń 72 cechy
    feature_vector = [] # znajduje się tu przekonwertowany patch z określoną liczbą poziomów jasności (64)
    for feature in feature_names:
        feature_vector.extend(list(greycoprops(glcm, feature).flatten()))
    return feature_vector


texture_folder = "Textures"
samples_folder = "TextureSamples"  # nazwa katalogu w jakim mają być zapisane stworzone próbki
paths = glob(texture_folder + "\\*\\*.jpg")  #lista wszystkich zdjęć - glob (\\*\\ - skrócony dostęp do plików - wildcards)

fil2 = [p.split(sep) for p in paths]  #nowa lista. podział ścieżki z paths na fragmenty katalog-podkatalog-plik
_, categories, files = zip(*fil2)

size = 128, 128  # krotka. rozmiar fragmentu - próbki 128 px
# wyłuskanie podkategori i plików z nowo powstałej krotki
features = [] # macierz do przechowaia wektora cech
for category, infile in zip(categories, files):
    img = Image.open(join(texture_folder, category, infile)) # wykorzystanie PIL
    xr = np.random.randint(0, img.width - size[0], 10)
    yr = np.random.randint(0, img.height - size[1], 10)  # losowanie 10 położeń próbek
    base_name, _ = splitext(infile)  # rozdzielenie - próbki nazwane identycznie jak zdjęcia (z indeksem)
    for i, (x, y) in enumerate(zip(xr, yr)):  #krotka zagnieżdżona. enumerate() zwraca indeks i item
        img_sample = img.crop((x, y, x + size[0], y + size[1])) # wycianie fragmentu -próbka
        img_sample.save(join(samples_folder, category, f'{base_name:s}_{i:02d}.jpg'))  # zapis do pliku. Nazwa plik: bazowa + indeks 02d - indeks zawsze dwucyfrowy
    #KONWERSJA DO SKALI SZAROŚCI#
        img_grey = img.convert('L')  # konwersja - szarość: symbol L
        feature_vector = get_glcm_feature_array(np.array(img_grey))  #wyliczany wektor cech
        feature_vector.append(category)  # dołączenie nazwy kategorii do wyliczonego wektora
        features.append(feature_vector)  # dołączenie wektora do wszystkich wektorów

full_feature_names = get_full_names()  #wyliczona lista cech
full_feature_names.append('Category')  #

df = DataFrame(data=features, columns=full_feature_names)  # zapis do pliku przy pomocy pandas
df.to_csv('textures_data.csv', sep=',', index=False) #index=false -  pominięcie indexu w wygenerowanych danych

#ODCZYT CECH I KLASYFIKACJA


from sklearn import svm  #obiekt klasyfikatora
from sklearn.model_selection import train_test_split # funkcja do podzielenia zbioru na bazowy i testowy
from sklearn.metrics import accuracy_score # accuracy - policzenie dokładności klasyfikatora


classifier = svm.SVC(gamma='auto') #zbudowany z radialna funkcją bazową - domyslne parametry
data = np.array(features) # zapis danych do tablicy array (tablica numpy)
x = (data[:, :-1]).astype('float64')    #wybór wszystkich wierszy bez ostatniej kolumny(kategorii). astype - zamiana typu
y = data[:, -1] #zapis etykiet

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33) # podane osobno wektory cech i kolumne etykiet kategorii

classifier.fit(x_train,y_train) #budowa klasyfikatora
y_pred = classifier.predict(x_test) #predykcja dla części testowej
acc = accuracy_score(y_test, y_pred)
print(acc)