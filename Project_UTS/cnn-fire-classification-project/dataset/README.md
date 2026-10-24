# Dataset

Dataset yang digunakan: Fire Recognition Image Dataset (Mendeley Data).

- URL: https://data.mendeley.com/datasets/7jk6xh7h6w/4
- DOI: https://doi.org/10.17632/7jk6xh7h6w.4
- Lisensi yang dicantumkan sumber: CC BY 4.0; periksa halaman sumber sebelum penggunaan/redistribusi.

Unduh dan ekstrak hanya arsip/folder **Original Images**. Jangan mencampur gambar original dengan gambar hasil augmentasi yang berasal dari gambar yang sama, karena dapat menyebabkan data leakage di antara training, validation, dan testing.

Letakkan empat folder kategori original di `data/raw/Original_Images/`:
- `Real Fire/`
- `Smoke/`
- `Safe Fire/`
- `Artificial Fire/`

Dataset tidak disertakan di repository untuk menjaga ukuran repository dan menghormati ketentuan sumber. Periksa lisensi dan atribusi dataset sebelum publikasi.
