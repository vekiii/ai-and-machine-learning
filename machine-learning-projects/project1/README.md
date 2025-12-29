# 📊 Analiza i klasifikacija morfoloških karakteristika oposuma

Ovaj projekat istražuje biološki skup podataka o oposumima kroz tri ključna ML problema: **klasifikaciju geografske regije**, **klasifikaciju pola** i **regresiju starosti**. Glavni fokus rada je na **interpretabilnosti modela (XAI)** i razumevanju razloga zašto modeli greše u specifičnim biološkim scenarijima.

## 🗂️ O skupu podataka
Skup podataka sadrži morfološke mere (dužine tela, glave, repa, širinu lobanje, itd.) jedinki oposuma prikupljenih sa različitih lokacija u Australiji. Podaci su skalirani korišćenjem `StandardScaler` pre procesa modelovanja.

---

## 🔍 Ključni nalazi i interpretabilnost

### 1. Klasifikacija regije (Uspeh)
* **Model:** Logistička regresija / Random Forest.
* **Rezultat:** Visoka preciznost i stabilnost.
* **XAI uvid (SHAP):** Analiza je otkrila da je **dužina uva (`earconch`)** najvažniji prediktor. Oposumi iz Viktorije imaju značajno drugačiju morfologiju uva u odnosu na ostale populacije, što omogućava modelu laku diferencijaciju.

### 2. Klasifikacija pola (Problem biološkog preklapanja)
* **Problem:** Model dostiže tačnost od svega **~65%**.
* **Lasso (L1) redukcija:** Korišćenje L1 penalizacije rezultovalo je svođenjem koeficijenata većine varijabli na nulu, što sugeriše da ne postoji jasan morfološki "potpis" za pol.
* **LIME analiza:** Detaljnim uvidom u pojedinačne greške (npr. mužjak klasifikovan kao ženka), uočeno je da individualne varijacije u veličini (grudi, stomak) često nadvladavaju polne razlike. 
* **Zaključak:** Poređenjem tačno i netačno klasifikovanih mužjaka, utvrđeno je da model donosi odluke na osnovu "šuma" u podacima, jer su morfološke mere polova previše slične.

### 3. Predviđanje starosti (Nizak signal)
* **Model:** Support Vector Regression (SVR).
* **Rezultat:** $R^2 \approx 0.15$.
* **Interpretabilnost:** SHAP vrednosti su visoko koncentrisane oko nule, što matematički potvrđuje da fizičke mere tela nisu pouzdan indikator starosti jedinke u ovom uzorku.

---

## 🛠️ Tehnologije i metode
* **Programski jezik:** Python
* **Biblioteke:** `pandas`, `numpy`, `scikit-learn`
* **Interpretabilnost (XAI):** `SHAP` (Globalna objašnjenja), `LIME` (Lokalna objašnjenja)
* **Redukcija dimenzionalnosti:** PCA (Principal Component Analysis), L1 (Lasso) regularizacija

---

## 📈 Finalni zaključak
Projekat demonstrira važnost **objašnjivog AI-a**. Umesto da se fokusiramo samo na niske metrike tačnosti kod pola i godina, upotrebom SHAP i LIME metoda uspeli smo da naučno dokažemo da je problem u **biološkom preklapanju karakteristika**, a ne u samoj arhitekturi modela.

---

### Kako pokrenuti projekt?
1. Klonirajte repozitorijum: `git clone [URL_REPOS_OVDE]`
2. Instalirajte zavisnosti: `pip install pandas sklearn shap lime matplotlib seaborn`
3. Pokrenite Jupyter Notebook: `jupyter notebook`
