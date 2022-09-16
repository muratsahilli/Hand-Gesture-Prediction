# Hand-Gesture-Prediction

### Please read the report

## Dosya açıklamaları: 
 > - pyqt.py : El hareketlerini yaparken etiketleyerek kaydeder. El hareketlerinden oluşan iki farklı veri seti kaydedildi. Bu veri setleri data06.txt ve datatest_07.txt
 > - data06.txt : pyqt.py kullanılarak kaydedilmiş ve model eğitmede kullanılan veri seti
 > - datatest_07.txt : pyqt.py kullanılarak kaydedilmiş ve eğitilen modellerin doğruluğunu test eden veri seti
 > - prediction.py : Makine öğrenmesi modellerini eğitip sonuçlarını karşılaştırır ve en iyi iki modeli kaydeder. Kaydedilen modeller daha sonra test için kullanılacak
 > - test_py.py : Kaydedilmiş modelleri kullanarak var olan veri seti veya canlı akan veri üzerinden sınıflandırma yapar.

## File explanations:
 > - pyqt.py : Records hand movements while tagging. Two different data sets consisting of hand movements were recorded. These datasets are data06.txt and datatest_07.txt
 > - data06.txt : The dataset saved using pyqt.py and used to train the model
 > - datatest_07.txt : The dataset saved using pyqt.py and used to test trained and saved models.
 > - prediction.py : It trains machine learning models and compares their results and saves the two best models. Saved models will be used for testing later
 > - test_py.py : Makes classification over existing data set or live streaming data using recorded models.
