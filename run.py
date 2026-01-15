import numpy as np
import cv2
import os
import random
import sys
import time
from stm_ai_runner import AiRunner

# --- AYARLAR ---
COM_PORT = 'COM6'  # Aygıt yöneticisinden teyit et
BAUD_RATE = 115200
DATA_FILE = "mnist.npz"  # İndirdiğimiz veri seti dosyası


def load_local_mnist():
    """TensorFlow kullanmadan numpy ile veriyi yükler"""
    if not os.path.exists(DATA_FILE):
        print(f"❌ {DATA_FILE} bulunamadı! 'dosyayi_indir.py' scriptini çalıştır.")
        sys.exit(1)

    print(f"📂 {DATA_FILE} yükleniyor...")
    with np.load(DATA_FILE, allow_pickle=True) as f:
        # Test verilerini alıyoruz
        x_test, y_test = f['x_test'], f['y_test']

    print(f"✅ Veri yüklendi. {len(x_test)} adet resim hazır.")
    return x_test, y_test


def print_ascii_art(image, label):
    """Resmi terminale çizer"""
    print(f"\n--- GÖNDERİLEN (Gerçek: {label}) ---")
    for row in image:
        line = ""
        for pix in row:
            if pix > 200:
                line += "##"
            elif pix > 50:
                line += ".."
            else:
                line += "  "
        print(line)
    print("-" * 28)


def run_cnn_test():
    # 1. Veriyi Hazırla
    x_test, y_test = load_local_mnist()

    runner = AiRunner()
    print(f"🔌 {COM_PORT} portuna bağlanılıyor...")

    # REFERANS KODUNDAKİ GİBİ BAĞLANTI
    if runner.connect('serial', port=COM_PORT, baudrate=BAUD_RATE):
        print("❌ HATA: Bağlantı kurulamadı!")
        return

    try:
        # 2. Model Keşfi (Referans kodun mantığı)
        names = runner._drv.discover()
        if not names:
            print("❌ Kartta model bulunamadı!")
            return
        model_name = names[0]
        print(f"✅ Bulunan Model: {model_name}")

        while True:
            # ---------------------------------------------------------
            # SEÇİM BÖLÜMÜ
            # ---------------------------------------------------------
            idx = random.randint(0, len(x_test) - 1)
            raw_img = x_test[idx]  # (28, 28) boyutunda ham resim
            actual_label = y_test[idx]

            # Ekrana çizelim
            print_ascii_art(raw_img, actual_label)

            # ---------------------------------------------------------
            # PREPROCESSING (Hu Moments yerine RESİM Hazırlama)
            # ---------------------------------------------------------
            # 1. Normalize et (0-255 -> 0.0-1.0)
            img_float = raw_img.astype(np.float32) / 255.0

            # 2. Boyut Ekleme (Model 28x28x1 bekliyor)
            # (28, 28) -> (28, 28, 1)
            img_input = np.expand_dims(img_float, axis=-1)

            # 3. Batch Ekleme (STM AI Runner list içinde batch bekler)
            # (28, 28, 1) -> (1, 28, 28, 1)
            input_data = np.expand_dims(img_input, axis=0)

            # Not: Eğer model 32x32 eğitildiyse burada cv2.resize gerekirdi.
            # Şu an optimize (28x28) modele göre yapıyoruz.

            # ---------------------------------------------------------
            # TAHMİN (Referans kodun aynısı)
            # ---------------------------------------------------------
            print("🚀 Karta gönderiliyor...", end="")
            start_t = time.time()

            # invoke_sample referans koddaki gibi kullanıldı
            outputs, profiler = runner._drv.invoke_sample([input_data], name=model_name)

            duration = (time.time() - start_t) * 1000
            print(f" ({duration:.1f} ms)")

            # ---------------------------------------------------------
            # SONUÇ İŞLEME
            # ---------------------------------------------------------
            if outputs:
                predictions = outputs[0].flatten()
                predicted_class = np.argmax(predictions)
                score = predictions[predicted_class]

                # Quantized model (int8) dönerse normalize et
                if score > 1.0: score /= 255.0

                print("\n" + "⭐" * 20)
                print(f" GERÇEK ETİKET      : {actual_label}")
                print(f" STM32 TAHMİNİ      : {predicted_class}")
                print(f" GÜVEN ORANI        : %{score * 100:.2f}")
                print("⭐" * 20)

                if actual_label != predicted_class:
                    print("⚠️  Tahmin yanlış!")
            else:
                print("⚠️  Çıktı alınamadı.")

            # Döngü kontrolü
            if input("\nDevam? [Enter] / Çıkış [q]: ").lower() == 'q':
                break

    except Exception as e:
        print(f"\nHata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.disconnect()
        print("🔌 Bağlantı kesildi.")


if __name__ == "__main__":
    run_cnn_test()





PYTHON KODU