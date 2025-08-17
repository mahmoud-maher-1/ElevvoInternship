# Named Entity Recognition (NER) Project

## Overview
This project focuses on **Named Entity Recognition (NER)**, a Natural Language Processing (NLP) task that automatically identifies entities such as **persons, organizations, and locations** within text. The project is built using the **CoNLL-2003 dataset**, which is a standard benchmark for evaluating NER systems.

The primary objective is to compare two distinct approaches:
1. **Model-Based NER** – leveraging pre-trained artificial intelligence models.  
2. **Rule-Based NER** – utilizing human-designed rules and dictionaries.

---

## Objectives
- Apply and evaluate **pre-trained AI models** for NER.  
- Design and test a **rule-based NER system** using dictionaries of names, organizations, and places.  
- Compare performance, strengths, and limitations of both methods.  
- Highlight the potential of hybrid approaches that combine AI and rule-based techniques.

---

## Methodology

### 1. Data Preparation
- The CoNLL-2003 dataset was used and divided into **training, validation, and test sets**. (Which is useful if needed to be used to train models such as CRF)  
- Sentences were **tokenized** for processing.  

### 2. Model-Based NER
- Implemented using **spaCy’s pre-trained models**:  
  - **Large transformer-based model (`en_core_web_trf`)**  
  - **Small lightweight model (`en_core_web_sm`)**  
- Entities were extracted and **visualized** directly in the text.  
- This approach required more computational power but provided higher accuracy and adaptability.  

### 3. Rule-Based NER
- Created **dictionaries** for:  
  - Organizations  
  - Cities, countries, and states  
  - First names  
- Developed rules to match words in text against these dictionaries.  
- Demonstrated that rule-based methods can work effectively for well-defined categories but lack flexibility.  

### 4. Comparison
- **Model-Based Approach:** More accurate, adaptive, and capable of generalizing beyond predefined dictionaries.  
- **Rule-Based Approach:** Simpler, faster, and interpretable but limited in scope.  
- **Hybrid Potential:** A combined system could balance **precision** (rules) and **adaptability** (models).  

---

## Key Findings
- AI-driven NER models outperform rule-based methods in handling real-world text variability.  
- Rule-based NER is effective in controlled scenarios but cannot handle unseen words or complex contexts.  
- Hybrid strategies could provide practical benefits by merging the strengths of both approaches.  

---

## Applications
The techniques developed in this project can be applied to real-world scenarios such as:
- **Business Intelligence** – extracting company and location names from reports.  
- **Healthcare** – identifying patient details, drug names, or medical terms.  
- **Customer Service** – detecting key entities in chat logs or support tickets.  
- **Legal and Compliance** – automatically recognizing sensitive entities in documents.  

---

## Conclusion
This project illustrates the value of **Named Entity Recognition** in transforming raw text into structured information. By comparing **model-based** and **rule-based** methods, it highlights both the **power of modern AI techniques** and the **simplicity of traditional approaches**.  

The findings suggest that **hybrid approaches** could yield the most reliable results, leveraging the precision of rules alongside the adaptability of machine learning.

---

## Technologies Used
- **Python**  
- **spaCy (en_core_web_trf, en_core_web_sm)**  
- **NLTK**  
- **Pandas**  
- **Custom dictionaries (CSV, JSON, TXT)**  

---

## Dataset
- **CoNLL-2003 Dataset** – A widely used dataset for evaluating Named Entity Recognition tasks, containing annotations for persons, organizations, locations, and miscellaneous entities.

---
