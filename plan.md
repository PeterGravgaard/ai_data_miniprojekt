Vi har fået denne opgave. Den skal vi løse baseret på vores undervisningsmateriale som du har modtaget:
Workshop 1: Manglende data
Workshoppen foregår som gruppearbejde i gruppeområdet.

Indhold
Opfølgning på opgaver fra første kursusgange.
Relationelle databaser
Evaluering af dataimputeringsmetoder.
Forelæser
Simon Dahl Jepsen og Jesper Rindom Jensen
Opgaver


Arbejde i grupper på første del af miniprojektet

Tilegnelse af et egnet datasæt til arbejde med manglende data (eksempler)

Identificer typer af datamangel

Implementer/anvend minimum 2 dataimputeringsmetoder til at håndtere manglende data og sammelign (f.eks. ved brug af classifier med/uden imputering)

Strukturér data i minimum 2 relationelle databaser (se eksempel fra Forelæsning 2)
Eksempler på data
https://www.kaggle.com/datasets/slmsshk/medical-students-dataset
https://www.kaggle.com/datasets/matinmahmoudi/sales-and-satisfaction
https://www.kaggle.com/datasets/redlineracer/nfl-combine-performance-data-2009-2019https://www.kaggle.com/datasets/yasserh/horse-survival-dataset



Hvad skal vi gøre step for step.

Vi har lavet en jupyter notebook.
AI 11.4.2025, 12.15.03
Okay, lad os bryde opgaven ned i konkrete trin baseret på jeres undervisningsmateriale fra Lektion 1-5. I skal bruge en Jupyter Notebook til at dokumentere jeres arbejde, kode og konklusioner for hvert trin.

Workshop 1: Manglende data - Step-by-Step Guide

Fase 0: Forberedelse og Setup (I Jupyter Notebook)

Importér Nødvendige Biblioteker: Start jeres notebook med at importere de biblioteker, I forventer at skulle bruge. Baseret på Lektion 1-5 vil det sandsynligvis inkludere:

pandas (til dataindlæsning og manipulation)
numpy (til numeriske operationer, ofte brugt sammen med pandas og scikit-learn)
matplotlib.pyplot og seaborn (til datavisualisering - Lektion 5)
sklearn.impute (til dataimputering, f.eks. SimpleImputer, KNNImputer - Lektion 3)
sklearn.model_selection (til opdeling i trænings- og testdata, f.eks. train_test_split)
sklearn.neighbors (eller en anden simpel klassifikator som LogisticRegression for evaluering - implicit fra ML-omtale i Lektion 1 & 3)
sklearn.metrics (til evaluering af klassifikator, f.eks. accuracy_score)
sqlite3 (til databaseinteraktion - Lektion 2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # Eksempel classifier
from sklearn.metrics import accuracy_score
import sqlite3
Fase 1: Tilegnelse og Indledende Analyse af Datasæt (Opgave 1)

Vælg et Datasæt: Vælg ét af de foreslåede datasæt fra Kaggle (eller et lignende, hvis I har et bedre egnet fra jeres miniprojekt). Sørg for, at det har manglende værdier. Download CSV-filen.

Reference: Lektion 1 (Kilder til data), Opgavebeskrivelsen.
Indlæs Data: Brug pandas til at indlæse datasættet i en DataFrame.

# Erstat 'sti/til/din/data.csv' med den faktiske sti
df = pd.read_csv('sti/til/din/data.csv')
Indledende Dataudforskning: Få et overblik over dataene:

Vis de første par rækker: df.head()
Få information om kolonner, datatyper og ikke-null værdier: df.info() (Dette er centralt for at se manglende data!)
Få deskriptiv statistik: df.describe() (for numeriske kolonner)
Tæl manglende værdier pr. kolonne: df.isnull().sum()
Reference: Implicit fra datahåndtering i flere lektioner, specifikt visualisering i Lektion 5 til at forstå data.
Fase 2: Identifikation af Datamangel (Opgave 2)

Analyser Manglende Værdier: Baseret på outputtet fra df.info() og df.isnull().sum(), identificer hvilke kolonner der har manglende værdier og hvor mange.

Visualisér Manglende Data (Valgfrit men anbefalet): Brug seaborn til at lave et heatmap over manglende værdier for at se mønstre.

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap af Manglende Værdier')
plt.show()
Reference: Lektion 5 (Visualisering).
Identificer Typer af Datamangel (MCAR, MAR, NMAR): Diskuter og argumenter for, hvilken type datamangel I formoder, der er tale om for de relevante kolonner. Dette kræver ofte domæneforståelse eller logisk ræsonnement baseret på de observerede data.

MCAR (Missing Completely At Random): Mangler data helt tilfældigt? Er der ingen sammenhæng mellem manglen og nogen af kolonnerne (hverken den manglende eller de observerede)? (Svært at bevise, ofte en standardantagelse hvis intet andet tyder på MAR/NMAR).
MAR (Missing At Random): Er sandsynligheden for at mangle data i kolonne A relateret til værdierne i en anden kolonne B (som er observeret)? (F.eks. mangler "indkomst" oftere for "yngre" respondenter?). I kan undersøge dette ved at gruppere data eller lave plots.
NMAR (Not Missing At Random): Er sandsynligheden for at mangle data i kolonne A relateret til den manglende værdi selv? (F.eks. folk med meget høj/lav indkomst undlader at svare?). (Sværest at identificere uden yderligere info).
Dokumentér jeres argumentation i notebook'en.
Reference: Lektion 3 (Slides, Forelæsningsnoter, Læsestof Kapitel 4).
Fase 3: Implementering og Sammenligning af Imputeringsmetoder (Opgave 3 & 4)

Forbered Data til Imputering:

Lav kopier af jeres DataFrame, så I ikke ændrer originalen: df_imputed1 = df.copy(), df_imputed2 = df.copy().
Identificer de kolonner, der skal imputeres. Vær opmærksom på, om de er numeriske eller kategoriske.
Vælg og Implementer Minimum 2 Imputeringsmetoder:

Metode 1 (f.eks. Mean/Median/Mode): Brug SimpleImputer fra scikit-learn. Vælg strategy='mean', strategy='median' (for numerisk data) eller strategy='most_frequent' (for kategorisk data).
  # For numerisk kolonne 'NumKolonne' med mean
  num_imputer = SimpleImputer(strategy='mean')
  df_imputed1['NumKolonne'] = num_imputer.fit_transform(df_imputed1[['NumKolonne']])

  # For kategorisk kolonne 'KatKolonne' med mode
  cat_imputer = SimpleImputer(strategy='most_frequent')
  df_imputed1['KatKolonne'] = cat_imputer.fit_transform(df_imputed1[['KatKolonne']])
  # Gentag for alle relevante kolonner i df_imputed1
Metode 2 (f.eks. KNN Imputation): Brug KNNImputer fra scikit-learn. Denne metode virker typisk bedst på numeriske data og bruger nabo-observationer til at estimere manglende værdier. Husk at vælge antal naboer (n_neighbors).
  # Antag at 'NumKolonne1' og 'NumKolonne2' er numeriske med manglende værdier
  knn_imputer = KNNImputer(n_neighbors=5) # Vælg passende k
  # KNNImputer kræver typisk at køre på flere kolonner samtidig
  cols_to_impute = ['NumKolonne1', 'NumKolonne2'] # Udskift med jeres kolonnenavne
  df_imputed2[cols_to_impute] = knn_imputer.fit_transform(df_imputed2[cols_to_impute])
  # Gentag for relevante *numeriske* kolonner i df_imputed2
Tjek for manglende værdier igen: df_imputed1.isnull().sum(), df_imputed2.isnull().sum() for at bekræfte imputeringen.
Reference: Lektion 3 (Slides, Forelæsningsnoter, Læsestof Kapitel 4, Python-værktøjer nævnt).
Evaluer og Sammenlign Imputeringsmetoderne: Den foreslåede metode er at bruge en klassifikator.

Vælg en Target Kolonne: Vælg en kategorisk kolonne i jeres datasæt, som I vil forsøge at forudsige (klassificere). Denne kolonne må ikke have manglende værdier efter imputering.
Vælg Features: Vælg de kolonner, I vil bruge som input (features) til klassifikatoren. Disse skal være numeriske (kategoriske features skal evt. konverteres til numeriske, f.eks. via one-hot encoding, men hold det simpelt i første omgang).
Forbered Data til Klassifikation:
Opret feature-matricer (X) og target-vektor (y) for:
Den originale DataFrame uden imputering (enten ved at droppe rækker med NaN eller droppe kolonner med NaN - diskuter konsekvenserne jf. Lektion 3). Lad os kalde dette X_original, y_original.
DataFrame efter Metode 1: X_imputed1, y_imputed1.
DataFrame efter Metode 2: X_imputed2, y_imputed2.
Vigtigt: Sørg for, at y er den samme (og komplet) i alle tilfælde. Fjern evt. rækker hvor y mangler før imputering. Håndter NaN i X for original data (f.eks. drop rækker/kolonner).
Håndter kategoriske features: Hvis I bruger features, der er kategoriske, skal de omdannes til numeriske. pd.get_dummies() er en simpel måde, men vær opmærksom på dimensionalitet.
Træn og Evaluer Klassifikator:
For hvert af de tre datasæt (original, imputed1, imputed2):
Split data i træning- og test-sæt: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Initialiser en simpel klassifikator: classifier = KNeighborsClassifier(n_neighbors=5) (eller LogisticRegression())
Træn klassifikatoren: classifier.fit(X_train, y_train)
Lav forudsigelser på test-sættet: y_pred = classifier.predict(X_test)
Beregn nøjagtigheden: accuracy = accuracy_score(y_test, y_pred)
Gem eller print nøjagtigheden for hvert datasæt.
Sammenlign Resultater: Sammenlign nøjagtigheden opnået med de forskellige imputationer (og den oprindelige/droppede version). Hvilken imputationsmetode gav den bedste performance for klassifikatoren? Diskuter hvorfor.
Visualiser Sammenligning (Valgfrit): Lav et simpelt søjlediagram, der viser nøjagtigheden for hver metode.
Reference: Lektion 3 (Evaluering af metoder), Lektion 5 (Visualisering), Lektion 1 (ML-koncepter).
Fase 4: Strukturering af Data i Relationelle Databaser (Opgave 5)

Design Database Skema: Baseret på jeres valgte datasæt (I kan bruge en af de imputerede versioner, f.eks. df_imputed1), design et skema med minimum 2 relationelle tabeller.

Identificer logiske grupperinger af kolonner. F.eks. for "Medical Students Dataset": en tabel Students (med ID, Age, Gender) og en tabel AcademicInfo (med ID, GPA, StudyYear, linked via StudentID).
Definer primærnøgler (unikke identifikatorer) for hver tabel.
Definer fremmednøgler for at skabe relationer mellem tabellerne (f.eks. StudentID i AcademicInfo refererer til ID i Students).
Dokumentér jeres skema i notebook'en (f.eks. med en beskrivelse eller et simpelt diagram).
Reference: Lektion 2 (Database design, Keys, Relationer, Eksempel: School DB).
Opret Database og Tabeller: Brug sqlite3 i Python til at oprette en databasefil og de tabeller, I har designet.

# Opret forbindelse (laver filen hvis den ikke findes)
conn = sqlite3.connect('min_database.db')
cursor = conn.cursor()

# Eksempel: Opret tabel 1 (tilpas til jeres design)
cursor.execute('''
CREATE TABLE IF NOT EXISTS Students (
    StudentID INTEGER PRIMARY KEY AUTOINCREMENT,
    Age INTEGER,
    Gender TEXT
    -- Tilføj flere kolonner fra jeres design
);
''')

# Eksempel: Opret tabel 2 (tilpas til jeres design)
cursor.execute('''
CREATE TABLE IF NOT EXISTS AcademicInfo (
    InfoID INTEGER PRIMARY KEY AUTOINCREMENT,
    StudentID INTEGER,
    GPA REAL,
    StudyYear INTEGER,
    -- Tilføj flere kolonner
    FOREIGN KEY (StudentID) REFERENCES Students (StudentID)
);
''')

conn.commit() # Gem ændringerne
Reference: Lektion 2 (Python Integration, SQL: CREATE TABLE, DDL).
Indsæt Data i Tabellerne: Iterér gennem jeres DataFrame (df_imputed1 eller den I har valgt) og indsæt data i de respektive SQL-tabeller.

# Vælg den imputerede dataframe I vil gemme
df_to_save = df_imputed1 # Eller df_imputed2

# Indsæt data i Students tabellen (eksempel)
for index, row in df_to_save.iterrows():
    cursor.execute('''
    INSERT INTO Students (Age, Gender)
    VALUES (?, ?)
    ''', (row['Age'], row['Gender'])) # Tilpas kolonnenavne
    # Få det sidst indsatte StudentID for at linke
    last_student_id = cursor.lastrowid# Indsæt data i AcademicInfo tabellen (eksempel)
cursor.execute('''
INSERT INTO AcademicInfo (StudentID, GPA, StudyYear)
VALUES (?, ?, ?)
''', (last_student_id, row['GPA'], row['StudyYear'])) # Tilpas kolonnenavne
conn.commit() # Gem alle inserts
Vigtigt: Koden ovenfor antager, at hver række i DataFrame svarer til én studerende og dens akademiske info. I skal muligvis tilpasse logikken, hvis jeres skema er anderledes (f.eks. hvis én studerende kan have flere rækker af akademisk info over tid). Det kan være mere effektivt at bruge executemany som vist i Lektion 2 læsestof, især for store datasets.
Verificer Data (Valgfrit men anbefalet): Kør en simpel SELECT query for at tjekke, om data er indsat korrekt.

cursor.execute("SELECT * FROM Students LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)

cursor.execute("SELECT s.Age, a.GPA FROM Students s JOIN AcademicInfo a ON s.StudentID = a.StudentID LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)
Reference: Lektion 2 (SQL: SELECT, JOIN, DML).
Luk Forbindelsen:
python conn.close()

Fase 5: Konklusion og Refleksion

Opsummér Fund: Skriv en kort konklusion i jeres notebook, der opsummerer:
Hvilket datasæt I brugte, og hvilke typer datamangel I fandt.
Hvilke imputeringsmetoder I anvendte, og hvordan de påvirkede klassifikationsresultaterne.
Hvordan I strukturerede dataene i de relationelle databaser.
Eventuelle udfordringer eller interessante observationer undervejs.
Husk at kommentere jeres kode grundigt og skrive forklaringer og argumentationer for jeres valg direkte i Jupyter Notebook'en. Held og lykke med workshoppen!