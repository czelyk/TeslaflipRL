# TeslaFlipRL — MVP doğrulama raporu

## 20 Ekim 2026 — itici gösterimi ve kapsamlı test düzeltmesi

İlk izleme kısayolu `--controller off` kullanıyordu; bu nedenle açılan gösterimde iticiler kasıtlı olarak kapalıydı. Araçta fiziksel kuvvet kanalları vardı, fakat görünür nozullar bulunmuyordu ve ilk doğrulama 13 kanalın tamamını gerçek araç üzerinde ayrı ayrı kapsamıyordu. Bu eksikler düzeltildi.

- Varsayılan izleme artık dengeleyici ve yakın takip kamerası kullanıyor.
- Gövdeye bağlı 13 görünür nozul ve ateşlenirken turuncu egzoz geometrisi eklendi. Görsel ekler kütlesiz ve çarpışmasızdır; kuvvet hattı boyunca dışarı çıkarılan uygulama noktaları `r × F` momentini korur.
- `Iticileri_Test_Et.cmd`: otomatik ölçüm ve 13 kanallı canlı ateşleme gösterimi.
- `Iticileri_Elle_Dene.cmd`: kanal/güç sürgüleri; SPACE basılıyken ateşleme, R ile test gövdesini sıfırlama.
- Gerçek 698 kg gövdede 13 kanalın tamamı sıfır, yarım ve tam güçte ölçüldü. Tam güçte 2.000 N × 0,05 s / 698 kg ≈ 0,1433 m/s; ölçümler yaklaşık 0,1431–0,1434 m/s aralığında. Ön/arka pitch, sağ/sol roll ve yanal kanallar yaw momentleri oluşturdu. Alt merkez kanalı kaldırma üretirken açısal hız sıfır kaldı.
- Her kanal döndürülmüş gövde üzerinde de test edildi. Ayrıca 13 ayrı tam rampalı bölümde her seferinde yalnızca seçilen kanalın gerçekten kuvvet ürettiği günlüğünden doğrulandı. Bu ateşleme denemelerinde iniş başarısı hedeflenmedi.
- Panel anlık N değerini ve biriken N·s değerini, günlüğü kanal başına tepe kuvveti ve toplam itkiyi gösteriyor.
- **63 otomatik test geçti.** Otomatik 3B tezgâh gösterimi 13 kanal için tamamlandı. Dengeleyiciyle yakın kamera ve canlı panel kontrolü de yapıldı.

Ölçüm verisi: `runs/thruster_test/thruster_report.json`. Görsel: `runs/thruster_test/nozzle_firing.png`. Elle klavye/fare girişi otomatik olarak sürülmedi; bu mod kullanıcı denemesi için eklendi.

---

19 Ekim 2026 · Windows · Python 3.12 · CPU

Proje dizini: `C:\Users\Ahmet\Desktop\TeslaflipRL`

Çalıştırılabilir MVP tamamlandı. Bağımlılıklar projeye özel `.runtime` ortamına kuruldu. `setup.ps1` mevcut ortam üzerinde başarıyla tamamlandı; ilk kurulumda PyBullet ikili ortamı da bu çalışma sırasında gerçekten oluşturuldu. Windows PyPI kaynak kurulumu C++ derleyicisi eksikliği nedeniyle başarısız oldu; conda-forge hazır PyBullet paketiyle çözüldü. Qt paneli ayrı işlemde çalıştırılıyor. NumPy/OpenBLAS paketi kullanılarak gözlenen OpenMP kitaplık çakışması giderildi. Desteklenmeyen çift OpenMP çalıştırma ayarı kullanılmadı.

## Doğrulanan sonuçlar

| Kontrol | Sonuç |
|---|---|
| Otomatik testler | **34 geçti** |
| Paket bağımlılık kontrolü | `pip check`: sorun yok |
| Gymnasium ortam sözleşmesi | Geçti |
| Tohumlanmış tekrar üretilebilirlik | Geçti |
| Soldan sağa yaklaşma, rampada gerçek teker temasları | Geçti |
| Üst uçtan sola/yukarı fiziksel ayrılma | Geçti; kalkışta `vx ≈ −10,75 m/s`, `vz ≈ +8,56 m/s` |
| Üretim adımlarında konum/hız sıfırlama yasağı | Geçti; test, bu API çağrılarını engelleyerek tam yaklaşma/kalkışı çalıştırıyor |
| Fiziksel itici torku ve gövdeyle dönen kuvvet yönü | Geçti |
| Yumuşak dört teker inişi, sert/ters/yan iniş cezaları | Kontrollü fizik testleri geçti |
| PPO kısa eğitim, checkpoint, model kaydı ve yeniden yükleme | Geçti |
| Kayıtlı modelle tam bölüm izleme | Geçti |
| Son modelden ve periyodik checkpoint'ten ek eğitim | Geçti; toplam adım sayacı korundu |
| Yerel 3B pencere + canlı grafik paneli | Geçti |
| Canlı PPO eğitiminde tüm bölümlerin gösterimi | 512 adımda 2 tam bölüm; ikisi de kaydedildi |
| Panel hız/duraklatma kontrol mantığı ve bölüm günlüğü | Geçti |
| Pencere kapatılınca durdurma yolu | Bağlantı kapanması ve ortam durumu otomatik testten geçti |

Son uçtan uca smoke testi: `runs/release_smoke/smoke_report.json`. Canlı eğitim denemesi: `runs/live_verified/`. Checkpoint'ten devam denemesi: `runs/checkpoint_resume_verify/`. Bağımlılık sürümleri: `installed_versions.json`.

## Başlatma

Masaüstündeki proje klasöründe `Egitimi_Baslat.cmd` dosyasına çift tıklayın. Bu, varsayılan olarak her eğitim bölümünü canlı 3B ekranda gösterir. Kontrolsüz fizik gösterimi için `Simulasyonu_Izle.cmd` dosyasını açın.

```powershell
cd C:\Users\Ahmet\Desktop\TeslaflipRL
.\run.cmd train.py --steps 200000 --run-dir runs\ilk-egitim
.\run.cmd watch.py --model runs\ilk-egitim\final_model.zip --episodes 5
```

Hız ve duraklatma düğmeleri canlı paneldedir. Daha hızlı penceresiz eğitim için `--headless` kullanın. `README.md` kurulum, fizik, ödül, gözlem, iticiler ve devam etme seçeneklerini Türkçe açıklar.

## Sınırlar

Kısa doğrulama eğitimleri başarılı iniş öğrenildiğini göstermiyor: canlı eğitim denemesinin iki inişi de sert iniş olarak sonuçlandı. Yumuşak iniş başarısı, kontrollü fizik testinde doğrulandı; bu, PPO'nun görevi çözdüğü anlamına gelmez. Uzun eğitim ve ayrı değerlendirme bölümleri gerekir. Başlangıç durumu dahil bütün bölümler rampanın solundaki zeminde başlar.

Referans görseller önceki konuşmadan okunarak incelendi. Boyutlar, kütleler ve itki sınırları örnek simülasyon parametreleridir; gerçek araç veya donanım ölçümü değildir. Rampa kesitleri ve dört teker gerçek çarpışma cisimleridir; şasi basit bir kutu modelidir. Askı sistemi, aerodinamik ve lastik deformasyonu bu MVP'ye dahil değildir.

PPO sürdürme, model/optimizer/adım sayacını yükler; yarım kalan fizik bölümünün birebir durumunu yüklemez. Görsel hız, fizik zaman adımını değiştirmez ve bilgisayarın çizim hızına bağlıdır. `.runtime` ve `work/` yerel kurulum dosyaları kaynak arşivine dahil edilmez; arşiv yeni yerde açılırsa `setup.ps1` çalıştırılır.

## Teknik başvuru kaynakları

Uygulama arayüzleri için [Gymnasium Env](https://gymnasium.farama.org/api/env/), kayıt akışı için [Stable-Baselines3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html), kuvvet/temas API'leri için [PyBullet Quickstart](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf) incelendi. Windows ikili paketi [conda-forge Bullet dağıtımından](https://github.com/conda-forge/bullet-feedstock) kuruldu.
