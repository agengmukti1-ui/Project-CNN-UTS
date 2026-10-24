# Laporan Project UTS Computer Vision

## Implementasi Convolutional Neural Network (CNN) untuk Klasifikasi Citra Kebakaran dan Non-Kebakaran

**Nama:** [Nama Mahasiswa]  
**Mata Kuliah:** Computer Vision  
**Program Studi:** [Program Studi]  
**Universitas:** [Nama Universitas]

## Abstrak

Project ini mengimplementasikan Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar sintetis ke dalam dua kelas, Fire dan Non-Fire. Dataset publik diperoleh dari Mendeley Data melalui script downloader, kemudian dibagi menjadi data training, validation, dan testing. Model dibangun menggunakan TensorFlow/Keras. Evaluasi dilakukan menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix. Nilai evaluasi serta kesimpulan akhir harus diisi berdasarkan hasil eksperimen aktual.

**Kata kunci:** CNN, Computer Vision, Fire Classification, Deep Learning.

## 1. Pendahuluan

### 1.1 Latar Belakang

Klasifikasi citra adalah salah satu tugas computer vision yang bertujuan mengelompokkan gambar berdasarkan karakteristik visual. CNN dipilih karena mampu mempelajari fitur spasial dari citra melalui lapisan konvolusi. Project ini menjadi implementasi pembelajaran klasifikasi biner pada dataset Fire dan Non-Fire sintetis.

### 1.2 Rumusan Masalah
1. Bagaimana menyiapkan dataset publik Fire dan Non-Fire secara otomatis?
2. Bagaimana membangun model CNN untuk klasifikasi biner?
3. Bagaimana hasil evaluasi model pada data testing?

### 1.3 Tujuan
1. Mengimplementasikan model CNN.
2. Mengotomatisasi download dan preprocessing dataset.
3. Mengevaluasi performa model pada data test.

## 2. Dasar Teori

### 2.1 Computer Vision
Computer vision merupakan bidang yang mengembangkan metode agar komputer dapat memproses dan memperoleh informasi dari citra atau video.

### 2.2 Convolutional Neural Network
CNN adalah jaringan saraf yang menggunakan operasi konvolusi untuk mempelajari fitur citra. Implementasi ini memakai Conv2D, Batch Normalization, Max Pooling, Global Average Pooling, Dropout, Dense, dan aktivasi sigmoid untuk klasifikasi biner.

### 2.3 Preprocessing
Gambar diubah menjadi 180 × 180 piksel dan nilai piksel dinormalisasi ke rentang 0–1. Data dibagi dengan rasio 70% training, 15% validation, dan 15% testing untuk masing-masing kelas. Augmentasi ringan hanya diterapkan pada training.

### 2.4 Evaluasi
Accuracy mengukur proporsi prediksi benar. Precision, recall, dan F1-score dilaporkan untuk kelas Fire, sementara confusion matrix menunjukkan jumlah prediksi benar dan salah per kelas.

## 3. Metodologi

### 3.1 Dataset

- Nama: *Synthetic Image fire and non-fire dataset*.
- Sumber: https://data.mendeley.com/datasets/mx33vrmxrv/1
- DOI: https://doi.org/10.17632/mx33vrmxrv.1
- Lisensi pada sumber: CC BY 4.0.
- Isi: 1.000 gambar sintetis hasil Stable Diffusion, 500 Fire dan 500 Non-Fire, resolusi 260 × 260 piksel.
- Ukuran arsip yang tercantum pada halaman sumber: sekitar 89,5 MB.

Dataset bersifat sintetis. Oleh karena itu, hasil tidak dapat langsung digeneralisasi ke foto nyata, CCTV, atau kondisi kebakaran lapangan.

### 3.2 Alur Eksperimen
1. Mengunduh dataset menggunakan `download_dataset.py`.
2. Memeriksa dan mengekstrak arsip ke `data/raw/`.
3. Membagi data menggunakan `src/prepare_dataset.py`.
4. Melatih CNN menggunakan `src/train.py`.
5. Mengevaluasi data test menggunakan `src/evaluate.py`.
6. Menguji prediksi satu gambar menggunakan `src/predict.py`.

### 3.3 Lingkungan Pengujian
- Sistem operasi: [Isi aktual]
- Python: [Isi aktual]
- TensorFlow: [Isi aktual]
- CPU/GPU: [Isi aktual]

## 4. Implementasi

Jelaskan struktur program, arsitektur CNN, optimizer Adam, loss binary cross-entropy, ukuran gambar 180 × 180, batch size 32, epoch maksimum 20, serta callback yang digunakan. Sesuaikan penjelasan dengan versi kode yang benar-benar dijalankan.

## 5. Hasil dan Pembahasan

### 5.1 Hasil Training
Masukkan grafik `results/training_history.png` setelah training berhasil.

### 5.2 Hasil Evaluasi
Isi tabel berdasarkan `results/metrics.json`.

| Metrik | Hasil aktual |
|---|---|
| Accuracy | Belum diukur |
| Precision (Fire) | Belum diukur |
| Recall (Fire) | Belum diukur |
| F1-score (Fire) | Belum diukur |

### 5.3 Confusion Matrix
Masukkan `results/confusion_matrix.png` setelah evaluasi.

### 5.4 Analisis Kesalahan
Pilih beberapa prediksi salah dan bahas karakteristik gambar yang berkontribusi. Jangan membuat klaim sebelum memeriksa hasil aktual.

### 5.5 Keterbatasan
Dataset terdiri dari gambar sintetis sehingga dapat memiliki pola visual yang berbeda dari foto nyata. Model juga belum divalidasi untuk penggunaan operasional atau keselamatan.

## 6. Kesimpulan dan Saran

Lengkapi setelah training dan evaluasi. Nyatakan hasil berdasarkan metrik testing yang benar-benar diperoleh. Pengembangan berikutnya dapat menguji dataset foto nyata yang berlisensi sesuai dan membandingkan arsitektur CNN.

## Daftar Pustaka
1. Tanim, S. A., Aurnob, A. R., & Zadid, T. H. (2025). *Synthetic Image fire and non-fire dataset*. Mendeley Data. https://doi.org/10.17632/mx33vrmxrv.1
2. TensorFlow Documentation. https://www.tensorflow.org/
3. Keras Documentation. https://keras.io/
4. Scikit-learn Documentation. https://scikit-learn.org/stable/
