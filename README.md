# TeslaFlipRL

TeslaFlipRL, yüksek bir ters C rampasından fırlayan dört tekerlekli bir aracı iticilerle dengelemeyi öğrenen, yerel bilgisayarda çalışan bir fizik ve pekiştirmeli öğrenme denemesidir. Proje adı dışında araç genel bir simülasyon modelidir. Gerçek Tesla donanımını temsil etmez ve referans videonun gerçekliği hakkında bir iddia taşımaz.

## Senaryo

Her bölüm (episode) araç zeminde **soldan sağa, +X yönünde** ilerlerken başlar. Araç rampanın alt girişine ulaşır, kavis boyunca yükselir ve üst uçtan **sola, −X yönünde** ayrılır. Hareket PyBullet temaslarıyla oluşur; rampadan çıkışta araca konum veya hız ataması yapılmaz. PPO, uçuş sırasında sınırlı iticileri kullanarak açısal hızı azaltmayı ve aracı dört teker üzerinde, düşük düşey hızla indirmeyi amaçlar.

Rampa X–Z düzlemindedir; Y aracın enine eksenidir. Merkez çizgisi `x = R sin(θ)`, `z = R (1 − cos(θ))` biçimindedir. Böylece alt girişte teğet sağa bakar; üstte `θ > π/2` olduğunda teğet sola ve yukarı bakar. Dünya yerçekimi −Z yönündedir.

## Kurulum (Windows PowerShell)

Windows kurulumu hazır PyBullet ikili paketi içeren, projeye özel bir Python 3.12 ortamı kullanır. Projeyi aşağıdaki dizinde açın:

```powershell
cd C:\Users\Ahmet\Desktop\TeslaflipRL
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` proje içine micromamba ve `.runtime` ortamını kurar. Windows'ta yalnızca `pip install -r requirements.txt` yeterli değildir: PyBullet için hazır Windows paketi conda-forge üzerinden kurulur. `environment.yml` bu ortamın tarifidir. Komutlar ortamı elle etkinleştirmeden `run.cmd` ile çalışır. CPU yeterlidir; uyumlu bir PyTorch kurulumu ile `--device cuda` kullanılabilir.

## Önce çalıştığını doğrulayın

**İticileri görmek ve tek tek denemek:** proje klasöründeki `Iticileri_Test_Et.cmd` dosyasını açın. 13 kanalın her biri önce gerçek araç gövdesinde ölçülür, ardından yakın 3B görünümde sırayla ateşlenir. Turkuaz parçalar nozullar, turuncu koniler çalışan egzoz jetleridir. Bu ayrı test tezgâhında yerçekimi ve teker motorları kapalıdır; PPO bölümü veya iniş başarısı testi değildir. Kanallar arasında gövde sıfırlanır.

`Iticileri_Elle_Dene.cmd` dosyası elle test açar: sağdaki **Kanal 0..12** ve **Güç yüzde** sürgülerini seçin; **SPACE tuşunu basılı tutunca ateşler**, **R** aracı sıfırlar. Her iki testte de ekranda kuvvet (N), hız ve açısal hız görünür.

`Simulasyonu_Izle.cmd` artık iticileri kullanan dengeleyici ve yakın takip kamerasıyla başlar. Eski kısayoldaki `--controller off` iticileri kapatıyordu. Normal rampalı bölümlerde rampaya yaklaşırken/tırmanırken iticiler kapalı, havada etkindir; panel bunu açıkça gösterir. Bölüm sonunda anlık itki sıfıra inse bile biriken itki ve kanal başına tepe kuvvetler günlüğe kaydedilir.

```powershell
.\run.cmd smoke_test.py
.\run.cmd -m pytest -q
.\run.cmd thruster_test.py --headless
.\run.cmd watch.py --controller thruster-test --episodes 13 --camera follow
.\run.cmd watch.py --controller off --episodes 2
```

Smoke testi kısa bir eğitim, model kaydı ve yeniden yükleme denemesi yapar. `--gui` ekleyerek görsel smoke testi çalıştırabilirsiniz. Kontrolsüz izleme, rampaya yaklaşmayı ve rampadan sola ayrılmayı gözlemlemek içindir; başarılı bir iniş beklenmez.

## Canlı eğitim

```powershell
.\run.cmd train.py --steps 200000 --run-dir runs\ilk-deneme
```

Varsayılan kip canlı PyBullet 3B penceresidir: **eğitimdeki her bölüm** bu pencerede ilerler. Hız `--speed 1` ile gerçek zamana yakın, `--speed 2` ile daha hızlı, `--speed 0.5` ile daha yavaş ayarlanır. Ortamın fizik zaman adımı aynı kalır; hız seçimi yalnızca gösterim beklemesini değiştirir. Ekrandaki bölüm sayacı, kümülatif ödül, durum, hız, açı ve itici seviyeleri uçuşu izlemeye yardımcı olur. Ayrı gösterge penceresi bölüm ödüllerini ve başarı oranını gösterir.

Gösterge penceresinde ¼×, ½×, 1×, 2×, 4×, 8× ve Sınırsız hız düğmeleri ile Duraklat/Devam et düğmesi bulunur. Başarı grafiği son 20 bölümün oranını gösterir. Fizik hesabı seçilen hıza yetişemediğinde görüntü doğal olarak daha yavaş ilerler.

```powershell
.\run.cmd train.py --steps 200000 --speed 2 --checkpoint-freq 10000 --seed 42 --device cpu
```

Canlı izleme eğitim hızını sınırlar. Tercihe bağlı hızlı, penceresiz eğitim:

```powershell
.\run.cmd train.py --steps 200000 --headless --run-dir runs\hizli
```

`--no-dashboard` grafik penceresini kapatır. `--n-steps 1024` ve `--batch-size 128` PPO'nun varsayılan toplama ve güncelleme boyutlarıdır. Mevcut seçeneklerin tamamı için `train.py --help` ve `watch.py --help` kullanın.

## Kayıt, sürdürme ve izleme

```powershell
.\run.cmd train.py --steps 200000 --resume runs\ilk-deneme\final_model.zip --run-dir runs\devam
.\run.cmd watch.py --model runs\ilk-deneme\final_model.zip --episodes 5 --speed 1
```

`--resume` kaydedilmiş PPO modelinden belirtilen sayıda **ek adım** eğitim başlatır. PPO bu sayıyı tam rollout uzunluğuna yuvarlayabilir. Periyodik dosyalar `checkpoints/`, son model `final_model.zip` adıyla çalışma dizininde tutulur. Bir modeli taşırken aynı çalışma dizinindeki `env_config.json` dosyasını da taşıyın: izleme ve sürdürme, fizik ayarlarını modelin yanından veya bir üst klasörden okur. Rastgele itkiyi veya basit bir dengeleme denetleyicisini görmek için:

Optimizer ve toplam adım sayacı yüklenir; fizik bölümü yeniden başlar. Fizik durumu veya rastgele sayı akışının bit düzeyinde devamı kaydedilmez. Bölüm kayıtları `episodes.jsonl` ve `episodes.csv`; canlı panel görüntüsü `dashboard.png` olarak saklanır. Aynı çalışma klasörüyle devam edildiğinde bölüm geçmişine eklenir. Windows'ta panel, 3B pencereyle birlikte yanıt verebilmesi için ayrı işlemde çalışır; kapanırken otomatik sonlandırılır.

```powershell
.\run.cmd watch.py --controller random --episodes 3
.\run.cmd watch.py --controller stabilize --episodes 3
```

Bu denetleyiciler eğitim gerektirmez. Basit dengeleyici PPO'nun yerini almaz ve iniş başarısı garanti etmez.

Her tamamlanan bölüm anında `episodes.jsonl` ve `episodes.csv` dosyalarına eklenir. Kayıtlarda ödül, başarı, sonuç kodu, adım sayısı ve temas öncesi düşey hız bulunur. `run.json` eğitim tohumu ve paket sürümlerini, `env_config.json` fizik ayarlarını saklar. Grafik penceresi açıkken normal kapanışta `dashboard.png` üretilir; penceresiz çalışmada CSV/JSONL kayıtları kullanılır. İzleme kayıtları varsayılan `runs/watch` dizinindedir; `--run-dir` ile değiştirilebilir.

## Fizik ve öğrenme tasarımı

PyBullet 3B rijit gövde dinamiğini ve rampa/zemin/teker temaslarını hesaplar. Gymnasium gözlem, eylem, ödül ve bölüm sonu arayüzünü sağlar. Stable-Baselines3 PPO sürekli itki seviyelerini öğrenir. Ön, arka, sağ, sol ve alt bölgelere yerleştirilen iticiler kuvveti kendi bağlantı noktalarında uygular; gövdenin dönmesi fiziksel `r × F` momentinden gelir. Her kanalın itki sınırı vardır. Rampada hız kazanma ile uçuş dengelemesi ayrı görevlerdir; PPO yalnızca uçuş sırasında iticileri etkinleştirir.

Varsayılan ayarda fizik 240 Hz, denetim 60 Hz çalışır. Gövde 650 kg, her teker 12 kg'dır. Rampa yarıçapı 9 m, üst açısı 2,55 radyandır; eğri 100 çarpışma parçasından oluşur. Araç başlangıçta yaklaşık 25 m/s hızla gelir; teker motorları yaklaşma ve rampada çalışır, uçuşta kapanır. Her sıfırlamada başlangıç hızı ve yanal konuma küçük, tohumlanabilir değişiklik eklenir. Parametreler `teslafliprl/config.py` içindedir.

13 eylem bileşeninin her biri `[-1, 1]` aralığındadır: **−1 kapalı, 0 yarım itki, +1 tam itki** anlamına gelir. Sıfır vektörü iticileri kapatmaz. Her itici en çok 2.000 N uygular. Konumlar ve kuvvet yönleri araç gövdesine bağlıdır; araç döndükçe dünyadaki itki yönleri de döner.

| Kanallar | Yerleşim ve etki |
|---|---|
| `front_up`, `front_down` | Ön uçta yukarı/aşağı kuvvet; pitch momenti |
| `rear_up`, `rear_down` | Arka uçta yukarı/aşağı kuvvet; karşı pitch momenti |
| `left_up`, `left_down` | Sol kenarda yukarı/aşağı kuvvet; roll momenti |
| `right_up`, `right_down` | Sağ kenarda yukarı/aşağı kuvvet; karşı roll momenti |
| `front_left`, `front_right` | Ön uçta yanal kuvvet; yaw momenti |
| `rear_left`, `rear_right` | Arka uçta yanal kuvvet; karşı yaw momenti |
| `bottom_lift` | Alt merkezde yukarı kuvvet; ağırlık merkezi doğrultusunda ek kaldırma |

34 bileşenli gözlem; normalize konum ve doğrusal hız (6), gövdenin ileri/yukarı birim vektörleri (6), açısal hız (3), dört uçuş evresi göstergesi (4), bölüm/denge süreleri (2) ve mevcut itki seviyelerinden (13) oluşur. Vektör yönleri, Euler açılarının çevrim sınırındaki kesintiyi önler. Görüntüdeki roll/pitch/yaw açıları yalnızca kullanıcı gösterimidir.

Ödül terimleri zaman adımıyla ölçeklenir; görüntüleme hızını değiştirmek ödülü değiştirmez:

| Terim | Amaç |
|---|---|
| Gövdenin dünya yukarısıyla hizası | Havada düzlüğü destekler; ters duruş negatif katkı verir |
| Açısal hızın karesi | Kontrolsüz dönüşü azaltır |
| Yere yaklaşırken yüksek düşey hız | Sert inişi önlemeyi destekler |
| Toplam itki ve itki değişimi | Yakıt tüketimini ve aşırı titreşen komutları azaltır |
| Rampadan ayrılma | Fiziksel fırlatma tamamlandığında +5 |
| Sonuç | Başarılı iniş +150; başarısızlık −70; süre dolması −50 |

Başarı için ilk temasın düşey hızı 4 m/s'yi aşmamalıdır. Sonrasında **dört teker birden**, yeterince düz gövde, 0,7 m/s'den düşük düşey hız ve 2,5 rad/s'den düşük açısal hız koşulları 0,20 saniye korunmalıdır. Ters/yan gövde teması, sert veya kararsız iniş ve alan dışına çıkış başarısızlıktır. Varsayılan bölüm süresi 14 saniyedir. Tam hesap ve sonuç kodları `teslafliprl/env.py` içindedir.

Bu ilk sürüm araştırma ve geliştirme içindir. Araç/rampa boyutları ile itici sınırları belirli bir gerçek araca göre kalibre edilmemiştir. Kısa smoke testi yalnızca sistemin çalışmasını denetler; PPO'nun iniş öğrendiğini göstermez. Başarı, eğitim süresi ve rastgele tohuma göre değişir. Eğitim eğrileri ve bağımsız izleme bölümleriyle değerlendirin.

## Dosyalar

- `train.py`: canlı veya penceresiz PPO eğitimi ve checkpoint.
- `watch.py`: kayıtlı model veya örnek denetleyiciyle izleme.
- `smoke_test.py`: kısa uçtan uca çalıştırma.
- `teslafliprl/`: fizik ortamı ve görsel izleme bileşenleri.
- `tests/`: ortam sözleşmesi, fizik yönleri ve davranış kontrolleri.
- `requirements.txt`: Python bağımlılıkları.
- `environment.yml`, `setup.ps1`, `run.cmd`: Windows için hazır fizik paketi ve yerel çalıştırıcı.
- `runs/`: üretilen modeller ve eğitim kayıtları; sürüm kontrolüne alınmaz.

## Sorun giderme

- **PyBullet kurulumu derleyici istiyor:** Windows'ta `setup.ps1` ile conda-forge paketini kurun. Bu proje, Windows PyPI kaynak paketini derlemeyi gerektirmez.
- **Grafik penceresi açılmıyor:** masaüstü oturumunda `watch.py` deneyin. Uzak/penceresiz oturum için `--headless` seçin.
- **Eğitim çok yavaş:** canlı izleme yerine `--headless` kullanın veya gösterim hızını artırın.
- **İlk bölümler düşüyor:** bu beklenen başlangıç davranışıdır. Birkaç yüz adımlık doğrulama eğitimi öğrenme başarısı ölçümü değildir.
- **Eğitimi durdurma:** terminalde Ctrl+C kullanın; üretilen son modelden `--resume` ile devam edin.
"# TeslaflipRL" 
