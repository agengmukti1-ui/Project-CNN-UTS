# Implementasi Convolutional Neural Network (CNN) untuk Klasifikasi Citra Kebakaran dan Non-Kebakaran

**Project UTS Computer Vision | Python | TensorFlow/Keras**

> Status: project siap dikembangkan dan dijalankan setelah dataset publik diunduh. Hasil evaluasi belum dicantumkan karena harus berasal dari eksperimen yang benar-benar dijalankan.

## Abstrak

Kebakaran dapat menimbulkan kerugian terhadap lingkungan, infrastruktur, dan keselamatan manusia. Salah satu pendekatan *computer vision* yang dapat dipelajari untuk mengenali pola visual kebakaran adalah klasifikasi citra menggunakan *Convolutional Neural Network* (CNN). Project ini mengimplementasikan CNN untuk mengklasifikasikan gambar ke dalam dua kategori, yaitu **Fire** dan **No Fire**. Dataset yang digunakan adalah [Fire Recognition Image Dataset](https://data.mendeley.com/datasets/7jk6xh7h6w/4) yang menyediakan empat kategori sumber: Real Fire, Smoke, Safe Fire, dan Artificial Fire. Untuk eksperimen biner, Real Fire dan Smoke dipetakan ke Fire, sedangkan Safe Fire dan Artificial Fire dipetakan ke No Fire. Tahapan project mencakup pemeriksaan dataset, pembagian data, preprocessing, pembangunan model, pelatihan, dan evaluasi menggunakan accuracy, precision, recall, F1-score, serta confusion matrix. Hasil aktual akan diisi setelah eksperimen selesai.

**Kata kunci:** Computer Vision, CNN, Deep Learning, Image Classification, Fire Classification.

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Computer vision memungkinkan komputer memproses dan mengenali informasi dari gambar. Salah satu penerapannya adalah klasifikasi citra, yaitu menentukan kategori gambar berdasarkan pola visual yang dipelajari dari data. Dalam konteks kebakaran, klasifikasi citra dapat digunakan sebagai eksperimen untuk membedakan gambar yang menunjukkan kebakaran atau asap dari gambar yang tidak termasuk kategori bahaya api nyata menurut label dataset.

*Convolutional Neural Network* (CNN) merupakan arsitektur deep learning yang banyak digunakan pada pengolahan citra. CNN mempelajari fitur visual melalui lapisan konvolusi dan operasi terkait, kemudian menggunakan fitur tersebut untuk menghasilkan prediksi kelas.

Project ini membangun baseline CNN dengan Python dan TensorFlow/Keras. Proses persiapan data, pelatihan, evaluasi, dan keterbatasan model didokumentasikan agar dapat ditelusuri melalui kode dan repository.

### 1.2 Rumusan Masalah

1. Bagaimana menyiapkan dataset citra publik untuk klasifikasi kebakaran dan non-kebakaran?
2. Bagaimana membangun dan melatih model CNN untuk tugas klasifikasi citra?
3. Bagaimana kinerja model berdasarkan metrik evaluasi klasifikasi?
4. Apa keterbatasan model dan dataset pada eksperimen ini?

### 1.3 Tujuan

1. Mengimplementasikan CNN untuk klasifikasi citra.
2. Melakukan pemeriksaan dataset, preprocessing, dan pembagian data.
3. Melatih model serta mengevaluasi performanya pada data uji.
4. Mendokumentasikan kode, hasil eksperimen, dan cara menjalankan project agar dapat direproduksi.

### 1.4 Manfaat

Project ini menjadi sarana pembelajaran implementasi CNN pada computer vision. Repository membantu pembaca mempelajari alur eksperimen klasifikasi citra dari persiapan dataset hingga evaluasi. Model ini bukan pengganti sistem alarm atau prosedur keselamatan kebakaran profesional.

---

## 2. Dasar Teori

### 2.1 Computer Vision

Computer vision adalah bidang yang mempelajari cara komputer memperoleh informasi dari gambar atau video. Klasifikasi citra merupakan salah satu tugasnya, yaitu memetakan gambar ke satu label atau kategori.

### 2.2 Convolutional Neural Network

CNN adalah arsitektur jaringan saraf yang dirancang untuk mempelajari pola spasial pada citra. Model baseline dalam project ini menggunakan lapisan konvolusi, aktivasi ReLU, max pooling, dropout, dan lapisan klasifikasi sigmoid untuk menghasilkan probabilitas dua kelas.

### 2.3 Preprocessing Citra

Preprocessing menyiapkan gambar agar dapat digunakan model. Project mengubah ukuran gambar menjadi 180 × 180 piksel dan menormalisasi nilai piksel ke rentang 0–1. Augmentasi ringan dilakukan pada data training melalui lapisan model; data validation dan testing tidak diberi augmentasi acak.

### 2.4 Pembagian Dataset

Dataset dipisahkan menjadi training, validation, dan testing. Script persiapan membagi setiap kategori sumber secara terpisah sebelum memetakannya ke label biner, sehingga proporsi kategori sumber dipertahankan di setiap split. Untuk mengurangi risiko kebocoran data, gunakan hanya gambar pada folder **Original Images** dari dataset sumber; jangan mencampurkan gambar asli dengan versi augmented yang diturunkan darinya.

### 2.5 Evaluasi Model

- **Accuracy:** proporsi prediksi benar dari seluruh prediksi.
- **Precision:** proporsi prediksi Fire yang benar.
- **Recall:** proporsi contoh Fire aktual yang berhasil dikenali.
- **F1-score:** rata-rata harmonik precision dan recall.
- **Confusion matrix:** jumlah prediksi benar dan salah untuk setiap kelas.

Metrik final dihitung pada data testing yang tidak digunakan untuk memperbarui bobot model.

---

## 3. Metodologi

### 3.1 Dataset

Dataset yang digunakan adalah **Fire Recognition Image Dataset**, tersedia di Mendeley Data:

- Dataset: https://data.mendeley.com/datasets/7jk6xh7h6w/4
- DOI: https://doi.org/10.17632/7jk6xh7h6w.4
- Artikel deskripsi dataset: https://doi.org/10.1016/j.dib.2026.112801
- Lisensi yang dicantumkan sumber: Creative Commons Attribution 4.0 International (CC BY 4.0).

Deskripsi sumber menyebut 1.112 gambar asli dan 6.672 gambar hasil augmentasi dengan empat kategori: Real Fire, Smoke, Safe Fire, dan Artificial Fire. Periksa halaman dataset dan lisensinya saat mengunduh; jumlah berkas lokal harus diverifikasi, bukan diasumsikan.

**Pemetaan kelas project:**

| Kategori asli | Label project |
|---|---|
| Real Fire | Fire |
| Smoke | Fire |
| Safe Fire | No Fire |
| Artificial Fire | No Fire |

Catatan: kategori Artificial Fire bisa memuat api simulasi atau dekoratif. Oleh karena itu, label No Fire berarti kategori tersebut bukan kategori bahaya api nyata menurut skema sumber, bukan berarti tidak ada visual menyerupai api.

### 3.2 Alat dan Pustaka

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa pemrograman |
| TensorFlow / Keras | Membangun dan melatih CNN |
| NumPy | Operasi numerik |
| Matplotlib | Grafik training dan confusion matrix |
| Scikit-learn | Metrik evaluasi |
| Pillow | Membaca dan memeriksa citra |
| Git dan GitHub | Pengelolaan versi dan publikasi |

### 3.3 Alur Eksperimen

1. Mengunduh dataset dari halaman resmi.
2. Mengekstrak arsip **Original Images** saja.
3. Menjalankan script persiapan untuk memeriksa kategori, lalu membuat split train/validation/test.
4. Melatih model CNN baseline.
5. Menyimpan model terbaik dan grafik training.
6. Mengevaluasi model pada data test.
7. Mencoba prediksi gambar baru.
8. Menganalisis hasil dan keterbatasan.
9. Memperbarui laporan dan artikel Medium berdasarkan hasil aktual.

### 3.4 Skenario Pengujian

| Skenario | Tujuan |
|---|---|
| Training | Memperbarui bobot model |
| Validation | Memantau training dan memilih model terbaik |
| Testing | Mengukur performa akhir pada data yang tidak digunakan saat training |
| Prediksi gambar baru | Mendemonstrasikan penggunaan model |

### 3.5 Lingkungan Pengujian

Project dirancang untuk Python 3.10 atau 3.11 di Windows menggunakan VS Code dan TensorFlow/Keras pada CPU. Versi pustaka dan spesifikasi perangkat aktual perlu dicatat setelah program dijalankan.

---

## 4. Implementasi

Kode dipisahkan berdasarkan fungsi:

- `src/prepare_dataset.py`: membagi gambar original menjadi train/validation/test.
- `src/train.py`: membangun dan melatih CNN.
- `src/evaluate.py`: menghitung metrik testing dan menyimpan confusion matrix.
- `src/predict.py`: memprediksi satu gambar baru.
- `results/`: grafik dan metrik yang dihasilkan program.
- `report/`: kerangka laporan UTS.

### 4.1 Keluaran Eksperimen

Setelah script dijalankan, keluaran yang diharapkan adalah model terbaik, grafik training, ringkasan metrik, confusion matrix, dan contoh prediksi. Keluaran tersebut belum dianggap tersedia sebelum program benar-benar berhasil dijalankan.

---

## 5. Hasil dan Pembahasan

Bagian ini harus diperbarui berdasarkan hasil eksperimen aktual. Jangan memasukkan angka performa sebelum training dan testing dilakukan.

### 5.1 Hasil Training

Tambahkan grafik accuracy dan loss training/validation yang dihasilkan oleh `src/train.py`. Gunakan grafik untuk memeriksa perkembangan model dan indikasi overfitting.

### 5.2 Evaluasi Klasifikasi

| Metrik | Hasil aktual |
|---|---|
| Accuracy | Belum diukur |
| Precision | Belum diukur |
| Recall | Belum diukur |
| F1-score | Belum diukur |

Tambahkan confusion matrix dari `results/confusion_matrix.png` setelah evaluasi berhasil.

### 5.3 Analisis Kesalahan

Periksa beberapa contoh prediksi yang salah. Bahas pola yang benar-benar ditemukan, misalnya asap, pencahayaan, api terkendali, atau objek yang menyerupai api.

### 5.4 Keterbatasan

Kinerja model dipengaruhi variasi gambar, kualitas label, komposisi kelas, dan potensi kemiripan antar gambar. Dataset sumber memiliki gambar original dan hasil augmentasi; project ini menggunakan original saja untuk mengurangi risiko kebocoran antara split. Hasil pada dataset publik tidak menjamin performa yang sama pada CCTV atau kondisi nyata.

---

## 6. Kesimpulan dan Saran

Tuliskan kesimpulan setelah eksperimen selesai. Ringkas performa model pada data testing, hasil analisis kesalahan, serta batasan yang ditemukan. Pengembangan selanjutnya dapat mencakup dataset eksternal, perbandingan arsitektur, dan pengujian pada kondisi gambar yang lebih beragam.

---

## 7. Instalasi dan Cara Menjalankan

### 7.1 Persyaratan

- Python 3.10 atau 3.11.
- Git.
- Dataset yang sudah diunduh dari sumber resmi.

### 7.2 Clone Repository

Ganti `USERNAME` dengan username GitHub pemilik repository:

```bash
git clone https://github.com/USERNAME/cnn-fire-classification.git
cd cnn-fire-classification
```

### 7.3 Buat Virtual Environment

Di PowerShell Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika menggunakan Python 3.11, sesuaikan perintah menjadi `py -3.11`. Jika PowerShell menolak aktivasi, gunakan Command Prompt dan jalankan `.venv\Scripts\activate.bat`.

### 7.4 Instal Dependensi

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 7.5 Siapkan Dataset

1. Buka https://data.mendeley.com/datasets/7jk6xh7h6w/4.
2. Unduh dataset dan baca lisensinya.
3. Ekstrak **Original_Images.zip** (atau folder arsip original dengan nama yang tersedia pada versi yang diunduh).
4. Letakkan folder kategori original di dalam `data/raw/Original_Images/`.
5. Pastikan folder kelas yang terdeteksi bernama `Real Fire`, `Smoke`, `Safe Fire`, dan `Artificial Fire` atau variasi kapitalisasi/spasi yang setara.

Struktur yang diharapkan:

```text
data/
└── raw/
    └── Original_Images/
        ├── Real Fire/
        ├── Smoke/
        ├── Safe Fire/
        └── Artificial Fire/
```

Jika arsip memiliki folder pembungkus tambahan, pastikan keempat folder kelas tetap berada di bawah path yang diberikan ke script.

### 7.6 Persiapan dan Training

```bash
python src/prepare_dataset.py --input_dir data/raw/Original_Images
python src/train.py
python src/evaluate.py
```

### 7.7 Prediksi Gambar Baru

```bash
python src/predict.py --image path/ke/gambar.jpg
```

Perintah di atas mengasumsikan model sudah berhasil dilatih dan tersedia di `models/best_model.keras`.

---

## 8. Struktur Repository

```text
cnn-fire-classification/
├── data/
│   └── raw/                  # Dataset lokal, tidak diunggah ke GitHub
├── notebooks/
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                   # Model hasil training, diabaikan Git
├── results/
├── report/
│   └── laporan_project.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 9. Dataset dan Daftar Pustaka

1. Saha, S., Sutradhar, H., Islam, M. A., & Matin, M. M. H. (2026). *Fire Recognition Image Dataset*. Mendeley Data. https://doi.org/10.17632/7jk6xh7h6w.4.
2. *An open-access fire detection image dataset for research in computer vision and safety monitoring*. Data in Brief, 66 (2026), 112801. https://doi.org/10.1016/j.dib.2026.112801.
3. TensorFlow. *TensorFlow Documentation*. https://www.tensorflow.org/.
4. Keras. *Keras Documentation*. https://keras.io/.
5. Scikit-learn. *Scikit-learn Documentation*. https://scikit-learn.org/stable/.

Verifikasi kembali metadata bibliografi dan lisensi pada halaman sumber sebelum pengumpulan atau publikasi.

---

## 10. Artikel dan Publikasi

- **GitHub Repository:** `https://github.com/USERNAME/cnn-fire-classification`
- **Artikel Medium:** tambahkan tautan setelah artikel dipublikasikan.

Artikel Medium akan menjelaskan latar belakang, dataset, arsitektur CNN, preprocessing, eksperimen, hasil evaluasi, keterbatasan, dan cara menjalankan kode. Semua hasil yang dicantumkan harus sesuai dengan keluaran eksperimen.

---

## Status Project

Project ini merupakan implementasi pembelajaran. Hasil evaluasi belum tersedia sampai training dan testing selesai dijalankan. README, laporan, dan artikel harus diperbarui agar sesuai dengan kode serta hasil aktual.

## Penulis

- **Nama:** [Nama Mahasiswa]
- **Mata Kuliah:** Computer Vision
- **Program Studi:** [Program Studi]
- **Universitas:** [Nama Universitas]
