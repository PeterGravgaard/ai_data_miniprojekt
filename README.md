# AI & Data Miniprojekt: Klimadata Analyse

## Projektbeskrivelse
Dette projekt fokuserer på analyse og behandling af daglige klimadata med henblik på at håndtere både manglende data og støj i tidsserierne. Vi arbejder med et omfattende klimadatasæt, der giver os mulighed for at implementere og sammenligne forskellige databehandlingsteknikker.

## Datasæt
Vi anvender [Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data), som indeholder følgende klimavariabler:
- Temperatur
- Luftfugtighed
- Vindretning
- Vindhastighed
- Generelle vejrforhold

## Del 1: Håndtering af Manglende Data

### Opgaver
1. Analyse af datamangel i klimadatasættet
   - Identificering af mønstre i manglende data
   - Kortlægning af forskellige typer datamangel
2. Implementering af dataimputeringsmetoder
   - Minimum 2 forskellige metoder (f.eks. mean imputation, KNN, eller regression)
   - Sammenligning af metodernes effektivitet på klimadata
3. Strukturering i relationelle databaser
   - Design af database-schema for klimadata
   - Implementation af minimum 2 forskellige databasestrukturer

## Del 2: Håndtering af Støjfulde Data

### Opgaver
1. Datavisualisering af klimatidsserier
   - Visualisering af temperaturudvikling
   - Analyse af sæsonmønstre
   - Identifikation af støjkilder
2. Støjanalyse
   - Vurdering af støjniveau i forskellige klimavariabler
   - Identifikation af outliers og anomalier
3. Implementering af præprocesseringsteknikker
   - Moving average for udjævning af daglige variationer
   - Frekvensdomænefiltrering for støjreduktion
4. Resultatsammenligning
   - Visualisering af data før og efter præprocessering
   - Evaluering af metodernes effektivitet

## Tekniske Krav
- Python-baseret implementering af databehandlingsmetoder
- Anvendelse af relevante biblioteker (pandas, numpy, sklearn)
- Visualiseringsværktøjer (matplotlib, seaborn)
- Dokumentation af alle databehandlingstrin
- Sammenlignende analyse af forskellige teknikker

## Undervisere
- Simon Dahl Jepsen
- Jesper Rindom Jensen

## Setup og Installation
```bash
# Installation af nødvendige Python-pakker vil blive tilføjet her
pip install pandas numpy matplotlib seaborn sklearn
```

## Dokumentation
Projektet dokumenteres løbende med:
- Jupyter notebooks med analyser
- Visualiseringer af resultater
- Metodebeskrivelser og evalueringer
- Kodeeksempler og implementeringsdetaljer