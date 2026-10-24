# Implementasi Convolutional Neural Network (CNN) untuk Klasifikasi Citra Kebakaran dan Non-Kebakaran

**Project UTS Computer Vision**

## Abstrak

Kebakaran merupakan salah satu peristiwa yang dapat menimbulkan kerugian terhadap lingkungan dan aktivitas manusia. Pengolahan citra berbasis *deep learning* dapat dimanfaatkan untuk mengidentifikasi karakteristik visual kebakaran secara otomatis. Project ini bertujuan mengimplementasikan *Convolutional Neural Network* (CNN) untuk mengklasifikasikan citra ke dalam dua kategori, yaitu kebakaran (*Fire*) dan non-kebakaran (*No Fire*).

Penelitian dilakukan melalui tahapan pengumpulan dataset publik, preprocessing citra, pembagian dataset, pembangunan model CNN, pelatihan, dan evaluasi. Kinerja model akan diukur menggunakan accuracy, precision, recall, F1-score, serta confusion matrix. Hasil eksperimen dan analisis akan ditambahkan setelah seluruh proses pelatihan dan pengujian selesai dilakukan.

**Kata kunci:** Computer Vision, CNN, Deep Learning, Image Classification, Fire Classification.

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Klasifikasi citra merupakan salah satu penerapan *computer vision* yang bertujuan mengenali kategori suatu gambar berdasarkan karakteristik visualnya. Dalam konteks kebakaran, model klasifikasi dapat dilatih untuk membedakan citra yang mengandung kebakaran dari citra yang tidak mengandung kebakaran.

*Convolutional Neural Network* (CNN) merupakan arsitektur *deep learning* yang banyak digunakan untuk pengolahan citra karena mampu mempelajari pola visual melalui operasi konvolusi dan proses ekstraksi fitur.

Project ini mengimplementasikan CNN menggunakan Python untuk mempelajari proses klasifikasi citra, mulai dari persiapan dataset hingga evaluasi model.

### 1.2 Rumusan Masalah

1. Bagaimana membangun model CNN untuk klasifikasi citra kebakaran dan non-kebakaran?
2. Bagaimana melakukan preprocessing dan pelatihan model CNN pada dataset citra?
3. Bagaimana kinerja model berdasarkan metrik evaluasi klasifikasi?

### 1.3 Tujuan

1. Mengimplementasikan CNN untuk klasifikasi citra.
2. Menerapkan preprocessing dan pembagian dataset.
3. Mengevaluasi kinerja model menggunakan metrik klasifikasi.
4. Mendokumentasikan proses eksperimen dalam repository GitHub.

### 1.4 Manfaat

Project ini diharapkan dapat menjadi sarana pembelajaran penerapan CNN pada computer vision serta memberikan contoh implementasi klasifikasi citra yang dapat dipelajari dan dikembangkan lebih lanjut.

---

## 2. Dasar Teori

### 2.1 Computer Vision

Computer vision merupakan bidang yang mempelajari metode agar komputer dapat memperoleh informasi dari gambar atau video.

### 2.2 Convolutional Neural Network

CNN merupakan arsitektur jaringan saraf yang dirancang untuk mempelajari fitur visual dari citra. Komponen yang akan digunakan dalam model meliputi convolutional layer, activation function, pooling layer, dan fully connected layer.

### 2.3 Preprocessing Citra

Preprocessing dilakukan untuk menyiapkan gambar sebelum dimasukkan ke model. Tahapan dapat mencakup penyesuaian ukuran gambar, normalisasi nilai piksel, serta pengaturan label kelas.

### 2.4 Evaluasi Model

Evaluasi akan menggunakan accuracy, precision, recall, F1-score, dan confusion matrix. Pemilihan metrik tersebut membantu menilai performa model dari beberapa sudut pandang, bukan hanya berdasarkan jumlah prediksi yang benar.

---

## 3. Metodologi

### 3.1 Dataset

Dataset akan menggunakan sumber publik yang relevan dengan klasifikasi citra kebakaran dan non-kebakaran. Informasi sumber, jumlah citra, distribusi kelas, dan lisensi penggunaan akan dicantumkan setelah dataset final diverifikasi.

### 3.2 Alat dan Pustaka

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa pemrograman |
| TensorFlow / Keras | Pembangunan dan pelatihan model CNN |
| NumPy | Operasi numerik |
| Matplotlib | Visualisasi hasil training |
| Scikit-learn | Evaluasi klasifikasi |
| Pillow | Membaca dan mengolah citra |
| Git dan GitHub | Pengelolaan versi dan publikasi source code |

### 3.3 Alur Penelitian

1. Menentukan dan mengunduh dataset.
2. Memeriksa kualitas serta distribusi kelas dataset.
3. Melakukan preprocessing citra.
4. Membagi dataset menjadi data training, validation, dan testing.
5. Membangun arsitektur CNN.
6. Melatih model dan memantau hasil training.
7. Mengevaluasi model menggunakan data testing.
8. Menguji prediksi pada gambar baru.
9. Mendokumentasikan hasil eksperimen.

### 3.4 Skenario Pengujian

Model akan diuji menggunakan data testing yang tidak digunakan untuk memperbarui bobot selama training. Hasil pengujian akan digunakan untuk menghitung metrik evaluasi dan mengidentifikasi kesalahan klasifikasi.

---

## 4. Implementasi

Implementasi akan dikembangkan menggunakan Python di Visual Studio Code. Kode program akan dipisahkan berdasarkan fungsinya agar lebih mudah dipahami, diuji, dan dikembangkan.

Rencana komponen program:

- `train.py`: preprocessing dan pelatihan model.
- `evaluate.py`: evaluasi kinerja model.
- `predict.py`: prediksi citra baru.
- `notebooks/`: eksperimen dan analisis data.
- `results/`: visualisasi dan hasil evaluasi.

Petunjuk instalasi dan eksekusi program akan ditambahkan setelah struktur kode final tersedia.

---

## 5. Hasil dan Pembahasan

Bagian ini akan dilengkapi berdasarkan hasil eksperimen aktual.

### 5.1 Hasil Training

Grafik accuracy dan loss pada data training serta validation akan digunakan untuk mengamati perkembangan pelatihan dan indikasi overfitting.

### 5.2 Evaluasi Klasifikasi

Hasil evaluasi akan menyajikan accuracy, precision, recall, F1-score, dan confusion matrix dari data testing.

### 5.3 Analisis Prediksi

Beberapa contoh prediksi yang benar dan salah akan dianalisis untuk memahami kemampuan serta keterbatasan model.

### 5.4 Keterbatasan

Pembahasan akan mencakup keterbatasan dataset, variasi kondisi gambar, potensi kesalahan klasifikasi, serta batasan penggunaan model.

---

## 6. Kesimpulan dan Pengembangan

Kesimpulan akan disusun setelah eksperimen selesai dilakukan dan hasil evaluasi dianalisis. Pengembangan selanjutnya dapat mencakup penambahan variasi dataset, penyesuaian arsitektur, dan pengembangan antarmuka prediksi.

Model klasifikasi ini hanya menghasilkan label gambar dan tidak menunjukkan lokasi api secara spesifik. Hasil prediksi juga tidak dimaksudkan sebagai pengganti sistem keselamatan kebakaran profesional.

---

## 7. Instalasi dan Cara Menjalankan

### Persyaratan

- Python yang kompatibel dengan versi TensorFlow pada `requirements.txt`.
- Git.
- Dataset yang sudah diunduh sesuai petunjuk.

### Instalasi

```bash
git clone https://github.com/USERNAME/cnn-fire-classification.git
cd cnn-fire-classification

python -m venv .venv
```

Aktifkan environment pada Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instal dependensi:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Perintah training dan prediksi akan ditambahkan sesuai nama file serta konfigurasi program yang sudah diuji.

---

## 8. Struktur Repository

```text
cnn-fire-classification/
├── dataset/
│   └── README.md
├── notebooks/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── results/
├── report/
├── requirements.txt
├── .gitignore
└── README.md
```

Struktur tersebut merupakan rancangan awal dan akan disesuaikan dengan file implementasi final.

---

## 9. Dataset dan Referensi

### Dataset

- Sumber dataset: akan dicantumkan setelah verifikasi dataset final.
- Lisensi: akan dicantumkan sesuai ketentuan sumber asli.

### Referensi Teknologi

1. [TensorFlow Documentation](https://www.tensorflow.org/)
2. [Keras Documentation](https://keras.io/)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

Referensi ilmiah yang digunakan dalam laporan akan ditambahkan setelah penelusuran dan verifikasi sumber.

---

## 10. Artikel dan Publikasi

**Artikel Medium:** Akan ditambahkan setelah artikel dipublikasikan.

**Repository GitHub:** `https://github.com/USERNAME/cnn-fire-classification`

---

## Status Project

Project ini masih dalam tahap persiapan dan pengembangan. Bagian implementasi, hasil evaluasi, kesimpulan, serta tautan publikasi akan diperbarui berdasarkan eksperimen yang telah dijalankan.

## Penulis

Nama: [Nama Mahasiswa]

Mata Kuliah: Computer Vision

Program Studi: [Program Studi]

Universitas: [Nama Universitas]
