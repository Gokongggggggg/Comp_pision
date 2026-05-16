# Overview

Analisis ini membahas perbandingan feature matching antara citra original dan citra hasil augmentasi menggunakan tiga descriptor klasik: SIFT, ORB, dan AKAZE. Augmentasi yang diuji adalah Gaussian blur, Gaussian noise, dan JPEG compression. Matching dilakukan menggunakan BFMatcher dan Lowe Ratio Test, sehingga nilai yang dibandingkan adalah jumlah good matches yang dianggap cukup kuat. Pada sample ini, hasil matching memberikan gambaran tentang bagaimana setiap descriptor merespons perubahan kualitas citra.

# Result Summary

Hasil good matches menunjukkan bahwa SIFT menghasilkan jumlah match tertinggi secara keseluruhan. Pada SIFT, Gaussian blur menghasilkan 26 good matches, Gaussian noise menghasilkan 20 good matches, dan JPEG compression menghasilkan 31 good matches. ORB menghasilkan jumlah match yang lebih rendah, yaitu 4 pada Gaussian blur, 4 pada Gaussian noise, dan 9 pada JPEG compression. AKAZE menghasilkan 2 good matches pada Gaussian blur, 3 pada Gaussian noise, dan 5 pada JPEG compression.

Secara umum, JPEG compression mempertahankan jumlah match paling tinggi pada semua metode. Hal ini terlihat pada SIFT dengan 31 matches, ORB dengan 9 matches, dan AKAZE dengan 5 matches. Sebaliknya, Gaussian blur dan Gaussian noise cenderung menghasilkan jumlah good matches yang lebih rendah.

# Descriptor Comparison

Pada sample ini, SIFT dapat diinterpretasikan sebagai descriptor yang paling stabil dalam konteks matching. Jumlah good matches SIFT jauh lebih tinggi dibandingkan ORB dan AKAZE pada semua jenis augmentasi. Kemungkinan hal ini terjadi karena SIFT menggunakan representasi local gradient yang lebih kaya dan descriptor berdimensi lebih besar, sehingga informasi lokal di sekitar keypoint dapat direpresentasikan dengan lebih detail.

ORB menghasilkan jumlah good matches yang lebih sedikit. Hal ini dapat dipahami karena ORB dirancang sebagai metode yang lebih cepat dan ringan, dengan descriptor biner. Descriptor biner efisien secara komputasi, tetapi pada sample ini tampaknya kurang mampu mempertahankan korespondensi sebanyak SIFT. AKAZE juga menghasilkan match yang rendah, meskipun masih menunjukkan beberapa korespondensi pada setiap augmentasi. Hasil ini tidak berarti ORB dan AKAZE selalu lebih buruk, tetapi menunjukkan bahwa pada sample ini dan dengan parameter default, keduanya menghasilkan lebih sedikit match yang lolos Lowe Ratio Test.

# Augmentation Impact

JPEG compression mempertahankan match paling banyak pada ketiga metode. Hasil ini menunjukkan bahwa artefak kompresi kemungkinan tidak sepenuhnya menghilangkan struktur lokal penting pada citra. Selama edge, tekstur utama, dan pola lokal masih cukup terlihat, descriptor masih dapat menemukan korespondensi antara citra original dan citra terkompresi.

Gaussian blur dan Gaussian noise tampak lebih kuat menurunkan kualitas matching. Gaussian blur menghaluskan detail frekuensi tinggi, sehingga beberapa corner atau tekstur lokal dapat melemah. Gaussian noise menambahkan perubahan intensitas acak pada level piksel, sehingga pola local gradient dapat menjadi kurang konsisten. Kedua kondisi ini dapat mengurangi stabilitas keypoint maupun kualitas descriptor.

# Interpretation

Hasil ini menunjukkan bahwa performa feature matching sangat dipengaruhi oleh hubungan antara jenis descriptor dan jenis distorsi citra. Descriptor yang lebih kaya, seperti SIFT, dapat memberikan informasi visual lokal yang lebih kuat untuk matching. Namun, descriptor yang lebih ringan seperti ORB dan AKAZE tetap berguna, terutama ketika efisiensi komputasi menjadi pertimbangan.

Dalam konteks deepfake detection, classical feature descriptors memberikan informasi visual yang berguna, terutama untuk melihat perubahan struktur lokal, robustness terhadap augmentasi, dan pola statistik descriptor. Namun, fitur ini kemungkinan belum cukup jika digunakan sendirian untuk deteksi deepfake yang sangat reliabel. Deepfake dapat memiliki artefak yang halus dan bervariasi, sehingga descriptor klasik sebaiknya dipahami sebagai salah satu sumber informasi, bukan satu-satunya dasar keputusan.

# Conclusion

Pada sample ini, SIFT menghasilkan jumlah good matches tertinggi dan dapat dianggap paling stabil dalam eksperimen matching. ORB dan AKAZE menghasilkan good matches yang lebih sedikit, kemungkinan karena representasi descriptor yang lebih ringan dan lebih sensitif terhadap perubahan lokal tertentu. JPEG compression mempertahankan match paling banyak pada semua metode, sedangkan Gaussian blur dan Gaussian noise lebih kuat menurunkan kualitas matching.

Secara keseluruhan, hasil ini mendukung penggunaan descriptor klasik untuk analisis visual dan studi robustness. Namun, untuk klasifikasi real vs fake yang lebih andal, descriptor statistics dan matching statistics sebaiknya dikombinasikan dengan evaluasi machine learning serta dataset yang cukup seimbang dan representatif.
