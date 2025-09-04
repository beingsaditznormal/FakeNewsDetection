# Fake News Detection con Word Embedding  

Questo progetto affronta il problema del rilevamento delle fake news attraverso l’applicazione di tecniche di **word embedding** e modelli di **machine learning** e **deep learning**. L’obiettivo è confrontare diverse metodologie di rappresentazione testuale e classificazione per individuare l’approccio più efficace.  

## 🚀 Obiettivi  
- Analizzare l’efficacia di **TF-IDF**, **Word2Vec** e **Transformers (SBERT)** nella rappresentazione del testo.  
- Addestrare e valutare diversi algoritmi di classificazione supervisionata (Naive Bayes, Logistic Regression, KNN, SVM, Decision Tree, Random Forest, MLP).  
- Confrontare i risultati utilizzando metriche standard come **Accuracy, Precision, Recall e F1-score**.  
- Sviluppare una **pipeline modulare e replicabile**, integrata con una dashboard interattiva in **Streamlit**.  

## 📊 Dataset  
Il dataset utilizzato è [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), che contiene circa **45.000 articoli** suddivisi in fake e real news. Ogni istanza include titolo, contenuto, argomento e data di pubblicazione.  

## 🏗️ Architettura  
Il sistema è organizzato come una pipeline composta da più fasi:  
1. **Importazione dati** con `ImportData.py` e archiviazione in **MongoDB**.  
2. **Preprocessing linguistico** con `PreProcessing.py` (pulizia testo, tokenizzazione, rimozione stopwords, lemmatizzazione).  
3. **Embedding** con TF-IDF, Word2Vec e SBERT.  
4. **Classificazione** tramite modelli di ML e DL.  
5. **Valutazione e visualizzazione** con report e dashboard in Streamlit.  

## ⚙️ Requisiti  
- Python 3.10+  
- Librerie principali:  
  - `scikit-learn`  
  - `gensim`  
  - `pandas`, `numpy`  
  - `nltk` e `spacy`  
  - `sentence-transformers`  
  - `streamlit`  
  - `pymongo`  

## ▶️ Esecuzione  
1. Clonare la repository:  
   ```bash
   git clone https://github.com/tuo-username/fake-news-detection.git
   cd fake-news-detection
   ```  
2. Installare i requisiti:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Avviare la pipeline:  
   ```bash
   python main.py
   ```  
4. Visualizzare i risultati con la dashboard:  
   ```bash
   streamlit run dashboard.py
   ```  

## 📈 Risultati  
- **TF-IDF** ha ottenuto le performance migliori con **Linear SVM (Accuracy = 0.9967, F1 = 0.9968)**.  
- **Word2Vec** ha mostrato buoni risultati ma inferiori senza fine-tuning (Accuracy ≈ 0.96).  
- **SBERT** ha raggiunto performance competitive (Accuracy ≈ 0.97) con maggiore capacità di generalizzazione.  

## 📝 Conclusioni  
Il progetto dimostra come TF-IDF, pur essendo una tecnica semplice, resti estremamente efficace nella classificazione supervisionata di fake news. Word2Vec e SBERT offrono invece una maggiore capacità di catturare relazioni semantiche e contestuali, rendendosi utili in scenari che richiedono generalizzazione.  
