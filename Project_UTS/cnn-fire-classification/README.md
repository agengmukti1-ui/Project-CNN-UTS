# Implementasi CNN untuk Klasifikasi Citra Kebakaran dan Non-Kebakaran

**Project UTS Computer Vision | Python | TensorFlow/Keras**

> Status: kode dan alur eksperimen disiapkan. Angka evaluasi belum diisi karena harus berasal dari training dan testing yang benar-benar dijalankan.

## Abstrak

Project ini mengimplementasikan *Convolutional Neural Network* (CNN) untuk mengklasifikasikan citra ke dalam dua kelas, yaitu **Fire** dan **Non-Fire**. Dataset diambil dari *Synthetic Image fire and non-fire dataset* di Mendeley Data. Dataset tersebut berisi 1.000 gambar sintetis yang dibuat menggunakan Stable Diffusion, terdiri dari 500 gambar Fire dan 500 gambar Non-Fire, dengan resolusi 260 × 260 piksel. Arsip publik yang tercantum di halaman dataset berukuran sekitar 89,5 MB. Program downloader memeriksa metadata ukuran sebelum mengunduh, kemudian mengekstrak gambar secara otomatis. Data dibagi menjadi train, validation, dan test; model CNN dilatih dan dievaluasi menggunakan accuracy, precision, recall, F1-score, serta confusion matrix. Hasil aktual ditambahkan setelah eksperimen dijalankan.

**Kata kunci:** CNN, Computer Vision, Deep Learning, Fire Classification, Image Classification.

## 1. Pendahuluan

Klasifikasi citra merupakan salah satu tugas *computer vision* yang memetakan gambar ke kategori tertentu. Dalam project ini, CNN digunakan sebagai eksperimen pembelajaran untuk membedakan citra sintetis yang diberi label Fire dan Non-Fire. CNN dipilih karena lapisan konvolusinya dapat mempelajari pola visual pada citra. Project ini ditujukan untuk pembelajaran dan bukan sistem keselamatan kebakaran siap pakai.

### Rumusan masalah
1. Bagaimana menyiapkan dataset publik untuk klasifikasi Fire dan Non-Fire?
2. Bagaimana membangun dan melatih model CNN sederhana?
3. Bagaimana mengevaluasi model pada data test yang terpisah?

### Tujuan
1. Mengimplementasikan baseline CNN dengan TensorFlow/Keras.
2. Mengotomatisasi pengunduhan dan pembagian dataset.
3. Mengevaluasi model dengan metrik klasifikasi standar.
4. Mendokumentasikan kode dan keterbatasan eksperimen.

## 2. Dataset

Sumber resmi: [Synthetic Image fire and non-fire dataset — Mendeley Data](https://data.mendeley.com/datasets/mx33vrmxrv/1)  
DOI: [10.17632/mx33vrmxrv.1](https://doi.org/10.17632/mx33vrmxrv.1)  
Lisensi yang dicantumkan: **CC BY 4.0**. Cantumkan atribusi dataset saat membagikan laporan atau hasil.

Menurut deskripsi sumber, dataset terdiri dari 1.000 gambar sintetis hasil Stable Diffusion: 500 kelas `Fire` dan 500 kelas `Non-Fire`, resolusi 260 × 260 piksel. Ukuran arsip yang tercantum pada halaman sumber adalah sekitar 89,5 MB. Karena seluruh gambar bersifat sintetis, hasil model tidak dapat dianggap sebagai bukti kinerja pada foto nyata, CCTV, atau situasi kebakaran lapangan.

## 3. Metodologi

1. **Download:** `download_dataset.py` memeriksa metadata arsip Mendeley dan menolak unduhan bila ukuran melebihi 120 MB.
2. **Ekstraksi:** arsip diekstrak ke `data/raw/` setelah validasi ZIP dan pemeriksaan path.
3. **Pembagian data:** `src/prepare_dataset.py` mencari folder Fire dan Non-Fire, lalu membagi setiap kelas dengan rasio 70% train, 15% validation, dan 15% test (seed 42).
4. **Preprocessing:** gambar diubah menjadi 180 × 180 piksel dan dinormalisasi ke rentang 0–1.
5. **Training:** CNN menggunakan convolution, batch normalization, max pooling, global average pooling, dropout, dense layer, dan sigmoid untuk klasifikasi biner. Augmentasi ringan hanya diterapkan pada training.
6. **Evaluasi:** accuracy, precision, recall, F1-score, classification report, dan confusion matrix dihitung pada test set.
7. **Prediksi:** model terlatih dapat digunakan untuk memprediksi satu gambar baru.

### Metrik evaluasi
- **Accuracy:** proporsi seluruh prediksi yang benar.
- **Precision (Fire):** proporsi prediksi Fire yang benar.
- **Recall (Fire):** proporsi citra Fire yang berhasil ditemukan.
- **F1-score (Fire):** rata-rata harmonik precision dan recall.
- **Confusion matrix:** ringkasan jumlah prediksi benar dan salah untuk setiap kelas.

## 4. Struktur Repository

```text
cnn-fire-classification/
├── download_dataset.py       # Unduh dataset otomatis dari Mendeley
├── run_pipeline.py           # Jalankan seluruh pipeline berurutan
├── data/                     # Dataset lokal (tidak diunggah ke GitHub)
├── src/
│   ├── prepare_dataset.py    # Split Fire/Non-Fire
│   ├── train.py              # Training CNN
│   ├── evaluate.py           # Evaluasi test set
│   └── predict.py             # Prediksi satu gambar
├── results/                  # Grafik dan metrik hasil eksperimen
├── models/                   # Model hasil training (lokal)
├── report/laporan_project.md # Template laporan UTS
├── MEDIUM_ARTICLE_OUTLINE.md
├── requirements.txt
└── README.md
```

## 5. Instalasi dan Cara Menjalankan (Windows / VS Code)

Gunakan Python 3.10 untuk kompatibilitas TensorFlow yang ditetapkan di `requirements.txt`.

### 5.1 Buat environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika `py -3.10` tidak tersedia, periksa versi Python dengan `py -0p` dan pasang Python 3.10 terlebih dahulu.

### 5.2 Unduh dataset otomatis

```powershell
python download_dataset.py
```

Dataset akan diunduh dari sumber publik dan diekstrak ke `data/raw/`. Arsip publik yang tercantum sekitar 89,5 MB; koneksi internet diperlukan. Script tidak mengunduh jika API Mendeley tidak menyediakan URL arsip atau ukuran arsip melebihi batas aman 120 MB.

### 5.3 Siapkan split data

```powershell
python src/prepare_dataset.py
```

Output berada di `data/processed/{train,val,test}/{Fire,Non-Fire}/`.

### 5.4 Training dan evaluasi

```powershell
python src/train.py
python src/evaluate.py
```

File hasil akan tersimpan di `models/` dan `results/`. Jangan menuliskan metrik ke laporan sebelum proses ini selesai.

### 5.5 Prediksi gambar

```powershell
python src/predict.py --image "D:\path\ke\gambar.jpg"
```

Gambar untuk demonstrasi boleh berupa gambar lokal, tetapi hasil prediksi tidak boleh ditafsirkan sebagai peringatan keselamatan yang tervalidasi.

### 5.6 (Opsional) Jalankan seluruh pipeline

Setelah dependensi terpasang, perintah berikut menjalankan download, split, training, dan evaluasi berurutan. Proses training dapat memakan waktu dan memerlukan koneksi internet untuk tahap download.

```powershell
python run_pipeline.py
```

## 6. Hasil dan Pembahasan

Bagian ini perlu dilengkapi setelah eksperimen:

| Metrik | Hasil |
|---|---|
| Accuracy | Belum diukur |
| Precision (Fire) | Belum diukur |
| Recall (Fire) | Belum diukur |
| F1-score (Fire) | Belum diukur |

Tambahkan `results/training_history.png`, `results/confusion_matrix.png`, dan ringkasan `results/metrics.json`. Analisis juga contoh kesalahan prediksi dan risiko model terlalu menyesuaikan karakter visual gambar sintetis.

## 7. Keterbatasan dan Etika Penggunaan

- Dataset sepenuhnya berisi gambar sintetis, bukan foto kejadian nyata.
- Hasil evaluasi hanya menggambarkan performa pada split dari dataset ini.
- Model belum divalidasi pada CCTV, video, gambar ponsel, atau lingkungan lapangan.
- Jangan gunakan model ini sebagai satu-satunya dasar keputusan darurat atau keselamatan.
- Atribusikan dataset sesuai lisensi CC BY 4.0.

## 8. Referensi

1. Tanim, S. A., Aurnob, A. R., & Zadid, T. H. (2025). *Synthetic Image fire and non-fire dataset*. Mendeley Data, Version 1. https://doi.org/10.17632/mx33vrmxrv.1
2. TensorFlow. *TensorFlow Documentation*. https://www.tensorflow.org/
3. Keras. *Keras Documentation*. https://keras.io/
4. Scikit-learn. *Scikit-learn Documentation*. https://scikit-learn.org/stable/

## 9. GitHub dan Artikel Medium

- GitHub: `https://github.com/USERNAME/cnn-fire-classification`
- Medium: tambahkan tautan setelah artikel dipublikasikan.

Perbarui laporan dan artikel dengan hasil aktual; jangan mengarang angka evaluasi.

## Status

Kode disiapkan untuk dijalankan secara lokal. Dataset belum diunduh dan model belum dilatih oleh repository ini. Hasil eksperimen baru tersedia setelah seluruh perintah berhasil dijalankan.
