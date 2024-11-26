<h1>Data</h1>
Data used for the implementation and testing

<h3>DisgeNET</h3>
Egy betegségek-gének kapcsolátát leíró adathalmaz.
Segítségével lehet a betegség-gén mátrixot létrehozni.
De csak a gének nevei szerepelnek benne, ezért szükség van egy kapcsoló táblára a gén nevek és gén azonosítok kapcsolatával.

Oszlopai:

- geneId: a gén azonosítója
- geneSymbol: a gén neve
- diseaseId: betegség azonosítója
- diseaseName: a betegség neve
- diseaseType: a betegség típusa
- diseaseClass: a betegség osztályba tartozása
- diseaseSemanticType
- score: a betegség és gén közötti kapcsolat erőssége
- EI: a gén-betegsé hitelességének száma
- YearInitial
- YearFinal
- NofPmids: hány PudMed cikkben hivatkoztak a gén-betegség kapcsolatra
- NofSnps source:

<h3>String</h3>
Fehérje-gén kapcsoló adathalmaz. Egy génhez több fehérje is tartozhat, ezért mi most a legerősebb kapcsolatut vesszük csak figyelembe.

Oszlopai:

- protein1: fehérje azonosító
- protein2: fehérje azonosító
- combined_score: kapcsolat erőssége 0-1000 skálán, érdemes 0-1 közé váltani a tartományt.

<h3>GTEx</h3>
A gén expressziós adatait tartalmazza, tehát hogy egy gén egy adott sejtszövetben mennyire aktív.
Ezeket az információkat a gráfban a csomópontok értékeiként használjuk fel.

Oszlopai:

- name: gén azonosítója, a .<szám> azt jelöli hanyadik frissitésen van túl a gén, a legnagyobb érték kell nekem (a legfrissebb)
- description:
- ...: a sejtszövetekben való aktivitás météke, ..-... közötti az értékük.

<h3>Gén név és gén azonosító kapcsolóadathalmaz</h3>
Arra használom amire a neve is utal, ezért csak két oszlop érdekel az adathalmazból.
Join esetén oda kell figyelni arra, hogy mivel nem közvetlen erre készítették, akár töbször is szerepelhet a kapcsolás, ezért ki kell majd dobni a dublán előfordulókat.

- Gene stable: a gén azonosítója
- Gene name: a gén neve
