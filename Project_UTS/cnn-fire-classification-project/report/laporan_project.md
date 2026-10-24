# Laporan Project UTS Computer Vision

## Implementasi Convolutional Neural Network (CNN) untuk Klasifikasi Citra Kebakaran dan Non-Kebakaran

**Nama:** Ageng Muktianto  
**Mata Kuliah:** Computer Vision  
**Program Studi:** Magister Teknik Informatika  
**Universitas:** Universitas Pamulang

## Abstrak
Project ini mengimplementasikan CNN untuk klasifikasi citra Fire dan No Fire menggunakan dataset publik Fire Recognition Image Dataset. Dataset dipersiapkan dengan menggunakan gambar original, kemudian dibagi menjadi training, validation, dan testing. Model dilatih menggunakan TensorFlow/Keras dan dievaluasi menggunakan accuracy, precision, recall, F1-score, serta confusion matrix. Hasil dan kesimpulan akhir harus ditambahkan berdasarkan eksperimen aktual.

**Kata kunci:** CNN, Computer Vision, Fire Classification, Deep Learning.

## 1. Pendahuluan

### 1.1 Latar Belakang
[Tulis latar belakang dengan sumber yang telah diverifikasi.]

### 1.2 Rumusan Masalah
1. Bagaimana membangun CNN untuk klasifikasi citra Fire dan No Fire?
2. Bagaimana proses preprocessing dan pembagian dataset?
3. Bagaimana hasil evaluasi model pada data testing?

### 1.3 Tujuan
1. Membangun model CNN.
2. Melatih dan mengevaluasi model.
3. Mendokumentasikan hasil eksperimen.

### 1.4 Manfaat
[Jelaskan manfaat pembelajaran dan batasan penggunaan model.]

## 2. Dasar Teori

### 2.1 Computer Vision
[Uraian teori dan referensi.]

### 2.2 Convolutional Neural Network
[Uraian convolution, activation, pooling, dropout, dan classification layer.]

### 2.3 Preprocessing
[Uraian resizing, normalisasi, augmentasi training, dan split data.]

### 2.4 Evaluasi
[Uraian accuracy, precision, recall, F1-score, dan confusion matrix.]

## 3. Metodologi

### 3.1 Dataset
Sumber: https://data.mendeley.com/datasets/7jk6xh7h6w/4  
Lisensi yang dicantumkan sumber: CC BY 4.0; verifikasi kembali sebelum publikasi.

Pemetaan kelas:
- Real Fire dan Smoke → Fire
- Safe Fire dan Artificial Fire → No Fire

Project menggunakan gambar original saja untuk mengurangi risiko kebocoran data akibat gambar augmentasi yang berasal dari gambar asli.

### 3.2 Alat dan Pustaka
Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, Pillow, Git, dan GitHub.

### 3.3 Alur Eksperimen
1. Unduh dan periksa dataset.
2. Split data original.
3. Latih CNN.
4. Evaluasi pada test split.
5. Analisis kesalahan.
6. Dokumentasikan hasil.

### 3.4 Lingkungan Pengujian
- Sistem operasi: [Isi aktual]
- Versi Python: [Isi aktual]
- TensorFlow: [Isi aktual]
- CPU/GPU: [Isi aktual]

## 4. Implementasi
Jelaskan struktur kode, arsitektur CNN, parameter training, batch size, epoch, optimizer, loss function, dan callback berdasarkan implementasi aktual.

## 5. Hasil dan Pembahasan

### 5.1 Hasil Training
[Masukkan grafik results/training_history.png.]

### 5.2 Hasil Evaluasi
Masukkan angka dari `results/metrics.json` dan `results/classification_report.txt`. Jangan mengisi hasil sebelum script evaluasi dijalankan.

| Metrik | Hasil aktual |
|---|---|
| Accuracy | [Isi hasil aktual] |
| Precision (Fire) | [Isi hasil aktual] |
| Recall (Fire) | [Isi hasil aktual] |
| F1-score (Fire) | [Isi hasil aktual] |

### 5.3 Confusion Matrix
[Masukkan results/confusion_matrix.png.]

### 5.4 Analisis Kesalahan dan Keterbatasan
[Analisis contoh prediksi yang salah, variasi gambar, kemungkinan bias dataset, dan batasan model.]

## 6. Kesimpulan dan Saran
[Tulis setelah eksperimen selesai. Jangan membuat klaim yang tidak didukung hasil.]

## Daftar Pustaka
1. Fire Recognition Image Dataset. Mendeley Data. https://doi.org/10.17632/7jk6xh7h6w.4.
2. TensorFlow Documentation. https://www.tensorflow.org/.
3. Keras Documentation. https://keras.io/.
4. Scikit-learn Documentation. https://scikit-learn.org/stable/.
