# Einleitung

Im Rahmen dieser Seminararbeit wurde ein Maschinelles Lernverfahren entwickelt, um basierend auf Datensätzen zu Radfahrten die zurückgelegten Höhenmeter vorherzusagen. Das Lernverfahren ist im zugehörigen Jupyter Notebook implementiert. Diese Seminararbeit dokumentiert und begründet das entwickelte Verfahren, begründet die gewählten Konfigurationen und analysiert die Lernergebnisse.

Diese Seminararbeit und das zugehörige Jupyter Notebook basiert auf dem individuellen Datensatz `myrides_01_Dataset_2.csv`.
Dieser enthält Daten von Radfahrten eines Radfahrers. Ein Datensatz (eine Fahrt) umfasst Datum, Start- und Enduhrzeit, Kilometer, Höhenmeter (gefahrene, niedrigste und höchste Höhenmeter über dem Meeresspiegel), Herzfrequenz, Geschwindigkeit und GPS-Koordinaten. Zudem beinhaltet jeder Datensatz Informationen, welches Fahrrad von welchem Typ genutzt wurde, sowie die Art der Fahrt und Fahrergewicht und Fahrergröße.
Auf Basis der gegebenen Daten sollen die gefahrenen Höhenmeter vorhergesagt werden (Regression).

# Lernverfahren

## Datenvorbereitung {#sec:data-preparation}

Die bereitgestellten Daten (CSV-Datei) werden zunächst in einen `pandas.DataFrame` eingelesen. Dabei wird `;` als Separator und `,` als Dezimaltrennzeichen verwendet. Der erste Schritt der Datenvorbereitung ist die Analyse auf Null-Werte bzw. fehlende Werte.
Diese zeigt, dass für etwa 93% der Datensätze `HR` und `maxHR` nicht gesetzt sind. Somit haben diese Eigenschaften wenig Mehrwert für das Lernverfahren und werden aus den Daten entfernt.

Im nächsten Schritt werden die Werte für Datum, Dauer und Start- und Endzeit in Zahlenwerte umgewandelt. Diese liegen zunächst als strukturierte Strings vor, welche nur schwer vom neuronalen Netz verwertet werden können. Dauer und Zeiten werden in Sekunden konvertiert, das Datum in Sekunden seit `1970-01-01` (Unixzeit).

Weiterhin müssen die kategorischen Daten der Radfahrten (Fahrradmodell, Radtyp, Fahrttyp) ebenfalls in Zahlenwerte umgewandelt werden. Dazu wird die Funktion `pandas.get_dummies` verwendet, welche kategorische Variablen in Indikatorvariablen konvertiert. Das bedeutet, dass zu jedem möglichen Wert der Spalte eine neue Spalte hinzugefügt wird, welche boolesche Werte (0 oder 1) enthält. Diese zeigen jeweils die zugehörige Kategorie an.
Nach diesen Vorbereitungsschritten liegen alle Eigenschaften der Fahrtdaten als Zahlenwerte vor. Dies ermöglicht, die Daten später einheitlich als Eingabewerte für das neuronale Netz zu verwenden.

\newpage

Als nächstes werden die vorbereiteten Daten zufällig in Trainingsdaten und Testdaten aufgeteilt. Beim Trainieren des neuronalen Netzes werden ausschließlich die Trainingsdaten verwendet. Das trainierte Modell wird abschließend anhand der Testdaten evaluiert. Sowohl Trainingsdaten als auch Testdaten werden in Labels (die zu lernende Spalte `Hm`) und Features (alle anderen Spalten) aufgeteilt.

Eine statistische Analyse der Daten zeigt, dass manche Datensätze starke Ausreißer enthalten, welches auf fehlerhafte Messungen hindeutet. Um das Training des neuronalen Netzes zu stabilisieren und Fehler durch mangelhafte Trainingsdaten zu vermeiden, werden vorher Ausreißer aus den Trainingsdaten herausgefiltert.
Dies wird erreicht, indem alle Zeilen aus dem Datensatz entfernt werden, die Spaltenwerte außerhalb des Intervalls $(\mu-4\sigma,\mu+4\sigma)$ der jeweiligen Spalte besitzen ($\mu$: Mittelwert, $\sigma$: Standardabweichung).
Durch dieses Verfahren werden starke Ausreißer aus dem Datensatz entfernt und das Netz wird stattdessen verstärkt auf Werte des erwarteten Normalbereiches trainiert. Selbst wenn die Datensätze mit solchen starken Ausreißern korrekt sind, hat es wenig Mehrwert das Netz mit diesen Daten zu trainieren, da nur sehr wenige Datenpunkte außerhalb dieser Spanne beim Verwenden des Netzes erwartet werden. Es bringt einen größeren Vorteil, das Netz mit hoher Genauigkeit auf die Mehrheit der erwarteten Datenpunkte zu trainieren, als die Gesamtheit der Werte nur mit verhältnismäßig schlechter Qualität vorhersagen zu können.

Die Normalisierung der Daten kann auch als Teil der Datenvorbereitung gesehen werden. In unserem Fall wird dies durch das neuronale Netz mittels einer Normalisierungsschicht umgesetzt, was in Abs. [-@sec:normalization] beschrieben wird.

## Feature Selection

Nach der Datenvorbereitung enthalten Trainings- und Testdatenset insgesamt 29 Spalten. Die Spalte `Hm` enthält die Zielwerte des Lernverfahrens (Labels), da die zurückgelegten Höhenmeter anhand der anderen Eigenschaften bestimmt werden sollen. Somit gibt es insgesamt 28 potenzielle Features.

Die Analyse der Daten zeigt, dass nicht alle Features gleichermaßen aussagekräftig für die Bestimmung der zurückgelegten Höhenmeter sind. Um nur Features zu nutzen, die einen Zusammenhang zu den Höhenmetern im Trainingsdatenset zeigen, werden Features basierend auf ihrer Korrelation zu `Hm` ausgewählt. Die Funktion `corr` für DataFrames berechnet paarweise die Korrelation der Spalten des DataFrames. Ist der Betrag des Korrelationskoeffizienten größer als $0,1$, wird die Spalte als Feature für das Modell ausgewählt.
Bei der Verwendung von `corr` wird der Pearson-Korrelationskoeffizient berechnet.

Die Feature Selection wird ausschließlich auf dem Trainingsdatenset durchgeführt, um zu vermeiden, dass Zusammenhänge des Testdatensatzes die Auswahl der Features beeinflussen. 
Es zu erkennen, dass bspw. zwischen `rider_height` und `Hm` kein Zusammenhang vorliegt (`nan`, da `rider_height` immer $1,83$ beträgt).

Da nur Features über einem bestimmten Korrelationskoeffizienten ausgewählt werden, wird die Spalte `rider_height` nicht ausgewählt. 
Stattdessen liegt bspw. eine hohe Korrelation zwischen `Hm`und `maxHm`vor. Daher wird `maxHm`als einer der Inputwerte für das Modell ausgewählt.

## Normalisierung {#sec:normalization}

Die Statistik für die einzelnen Spalten des Trainingsdatensatzes (`train_dataset.describe()`) zeigt, dass die Werte der verschiedenen Features deutlich unterschiedliche Größenordnungen aufweisen. Da die Features später mit den Gewichten des Modells multipliziert werden, beeinflusst die Größenordnung der Feautures signifikant die Größenordnung der Ausgabewerte und der Gradienten während des Lernprozesses. Um den Lernprozess zu stabilisieren und Schwankungen durch starke Abweichungen in den Größenordnungen zu vermeiden, werden die Daten zunächst normalisiert.

Hierzu wird ein Preprocessing-Layer von Tensorflow verwendet, welches als erste Schicht des neuronalen Netzes eingesetzt wird. Diese Schicht enthält keine lernbaren Parameter, wird also beim Trainieren des Netzes nicht verändert.
Allerdings wird vor dem Trainieren des Modells die Normalisierungsschicht auf die Trainingsdaten mit `normalizer.adapt()` angepasst. Diese Operation berechnet für jede Spalte der Trainingsdaten sowohl den Mittelwert (arithmetisches Mittel) als auch die Varianz und speichert diese Werte im inneren Zustand der Schicht.
Danach wird beim Verwenden der Schicht jedes Feature individuell normalisiert. Dazu werden die zuvor berechneten Mittelwerte verwendet, um den jeweiligen Z-Score (Standard-Score) zu berechnen. Diese Werte werden danach als normalisierte Eingabewerte im neuronalen Netz verwendet.

$$
f_{norm} = \frac{f - \mu}{\sqrt v}
$$

Der Einsatz einer solchen Normalisierungsschicht hat den Vorteil, dass die Normalisierung gleichermaßen beim Trainieren sowie bei der Verwendung des trainierten Netzes angewendet wird, z. B. bei der Validierung anhand der Testdaten. Bei der Verwendung des Netzes müssen die Daten somit nicht mehr separat normalisiert werden. [@TensorFlow2021]

## Multilayer Perceptron (Deep Neural Network)

Das Multilayer Perceptron wird in Tensorflow als `keras.Sequential` Model modelliert.
Um verschiedenen Konfigurationen des Netzes vergleichen zu können, werden Definition und Training in einer Hilfsfunktion ausgeführt. Über Parameter können dabei Anzahl und Größe der verborgenen Schichten konfiguriert werden sowie die zu verwendende Verlustfunktion. Der grundlegende Aufbau und alle anderen Parameter bleiben jedoch bei den verglichenen Netzen gleich.

\newpage

In diesem Abschnitt werden zunächst die gemeinsamen Eigenschaften der trainierten Netze beschrieben. Anschließend werden verschiedene Konfigurationen (verborgene Schichten, Verlustfunktion) verglichen, um eine geeignete Konfiguration für die Aufgabenstellung auszuwählen.

### Schichten

Alle Modelle enthalten als erste Schicht die zuvor konfigurierte Normalisierungsschicht. Auf diese folgen eine oder mehrere verborgene Schichten entsprechend des Parameters `hidden_layers`.
Für die verborgenen Schichten wird `layers.Dense` verwendet, in welchen jedes Neuron mit jedem Neuron aus der vorherigen Schicht verbunden wird.

Als Aktivierungsfunktion der verborgenen Schichten (Parameter `activation`) wird ReLu (Rectified Linear Unit) verwendet. Diese Funktion gibt alle positiven Werte weiter, negative Werte werden auf 0 gesetzt. Für positive Werte verhält sich ReLu linear.
Bei der Wahl der Aktivierungsfunktion spielt das Problem des verschwindenden Gradienten eine Rolle. Dieses kann bei tanh- und logistischen Aktivierungsfunktionen auftreten und führt zu einer sehr langsamen Ermittlung der Gewichtungen in der Trainingsphase, wenn die Gradienten sehr nah bei 0 liegen. ReLu ist von diesem Problem nicht betroffen, da die Ableitung der ReLu nach Eingabe positiver Eingabewerte 1 ergibt.
Der Vorteil von ReLu ist eine schnellere Berechnung, zudem ist die Funktion nicht vom Problem des verschwindenden Gradienten betroffen. [@Raschka2017, S.449-450]

Als letzte Schicht (Ausgabeschicht) wird `layers.Dense` mit einem einzelnen Neuron verwendet. Dieses Ausgabeneuron enthält den vorhergesagten Wert der gefahrenen Höhenmeter basierend auf den gegebenen Inputs.

### Training

Nach der Definition der Schichten des Modells wird der Lernvorgang durch Auswahl einer Verlustfunktion, eines Optimierers und der zu überwachenden Kennzahlen mit der Methode `compile()` konfiguriert.
Über den Parameter `loss` wird die Verlustfunktion gesetzt, mit der das neuronale Netz die Leistung der Trainingsdaten beurteilt. Der Wert der Verlustfunktion wird beim Training minimiert.

Der Optimierer (Parameter `optimizer`) passt die lernbaren Parameter des Netzes basierend auf den bekannten Daten und den Ergebnissen der Verlustfunktion an. Er legt fest, wie der Gradient der Verlustfunktion zur Aktualisierung der Parameter verwendet wird [@Chollet2018, S.183].
In diesem Programmentwurf wird jeweils der Adam-Algorithmus verwendet (`tf.keras.optimizers.Adam`) [@TensorFlow2021]. Der Vergleich von verschiedenen Optimierern ist außerhalb des Rahmens dieser Seminararbeit.
Der Optimierer wird hier mit einer abnehmenden Lernrate (`keras.optimizers.schedules.ExponentialDecay`) konfiguriert, die bei $0,0005$ beginnt und mit einem Faktor von $0,96$ abnimmt. Die abnehmende Lernrate sorgt für einen zügigen Lernfortschritt zu Beginn des Trainings, stabilisiert jedoch den weiteren Verlauf und minimiert das Risiko von Overfitting. Weiterhin wird das Modell so konfiguriert, dass während des Trainings und bei der Evaluation verschiedene Metriken neben der verwendeten Verlustfunktion aufgezeichnet werden.

Beim Ausführen der Methode `fit()` werden jeweils die Trainingsfeatures und Trainingslabels angegeben. Der Parameter `epochs` gibt die Anzahl der Iterationen über die Trainingsfeatures und Trainingslabels an, in diesem Fall beträgt der Wert $500$.

Der Parameter `validation_split` bestimmt, dass 20% der Trainingsdaten beim Trainieren des Models lediglich zur Validierung verwendet werden und nicht zum eigentlichen Training. Beim Lernvorgang wird die Verlustfunktion dann jeweils für die Trainingsdaten und die Validierungsdaten separat bestimmt. Dadurch kann beim Training schon frühzeitig ein Overfitting erkannt werden.
Ist der Verlust der Trainingsdaten z. B. deutlich geringer ist als der Verlust der Validierungsdaten, liegt ein Overfitting vor. Das bedeutet, dass das Modell Muster erlernt, die speziell in den Trainingsdaten vorhanden sind. Wird ein anderer Datensatz verwendet, sind diese Muster nicht relevant und führen zu Fehlern in den Vorhersagen.

Overfitting kann unter anderem durch das Verkleinern des Modells verhindert werden. Ein Ansatz für die korrekte Modellierung ist, verschiedene Architekturen anhand der Validierungsdaten zu bewerten. Zu Beginn sollten wenige Schichten und Parameter gewählt werden, die dann so lange erhöht werden, bis sich die Testergebnisse verschlechtert. [@Chollet2018, S.146]

### Vergleich von Anzahl und Größen der Schichten

Um eine geeignete Anzahl und Größe der Schichten des Netzes für das gegebene Lernproblem auszuwählen, werden die folgenden vier Modelle verglichen:

* Eine verborgene Schicht (Neuronenanzahl: 64)
* Zwei verborgene Schichten (Neuronenanzahl: 32, 16)
* Zwei breite verborgene Schichten (Neuronenanzahl: 64, 64)
* Drei breite verborgene Schichten (Neuronenanzahl: 128, 64, 32)

Bei der Betrachtung der Verlustfunktion für die Trainings- und die Validierungsdaten über die trainierten Epochen hinweg ist zu erkennen, dass die Modelle mit einer oder zwei (geringere Größe) verborgenen Schichten ein stabiles Ergebnis liefern. Die Verlustwerte unterscheiden sich hier nur geringfügig.

Die Modelle mit mehr bzw. größeren verborgenen Schichten liefern zwar einen geringeren Fehler für die Trainingsdaten als die kleineren Modelle. Allerdings liegt ein Overfitting vor. Das ist an der großen Differenz zwischen `loss` und `val_loss` erkennbar. Die Ausschläge im Graphen `three_hidden_layers` zeigen, dass das Training instabil ist.


### Vergleich von Verlustfunktionen

Da mit zwei verborgenen Schichten (Neuronenanzahl 32, 16) im vorangegangenen Vergleich das stabilste und beste Ergebnis erzielt wurde, wird das Modell `two_hidden_layers` für die weitere Betrachtung verwendet. Das Ziel ist die Identifikation einer geeigneten Verlustfunktion für das Modell.
Folgende Verlustfunktionen eignen sich im Allgemeinen für die Anwendung bei Regressionsproblemen:

- Mean Absolute Error
- Mean Absolute Percentage Error
- Mean Squared Error
- Mean Squared Logithmic Error

Diese Verlustfunktionen werden nun auf das Modell `two_hidden_layers` angewendet und die Lernergebnisse verglichen.
Die Graphen zeigen, dass das Training trotz verschiedener Verlustfunktionen weiterhin stabil ist und kein deutliches Over- oder Underfitting auftritt. Der Vergleich der Metriken zeigt, dass das Modell mit der Verlustfunktion `mean_absolute_error` die besten Verlustwerte für die Validierungsdaten erzielt.

# Bewertung des Lernergebnisses

Die vorherigen Vergleiche haben gezeigt, dass für die Lernaufgabe und die gegebenen Trainingsdaten ein Modell mit zwei schmalen verborgenen Schichten und `mean_absolute_error` als Verlustfunktion die besten und stabilsten Lernergebnisse erzielen kann, ohne dass ein Overfitting entsteht.
Deshalb wird das Modell `two_hidden_layers` ausgewählt, um das Lernverfahren anhand der Testdaten zu bewerten.
Für die Bewertung des Lernergebnisses wird zunächst das Modell auf die Testdaten angewendet, um die rohen Vorhersagen des Modells zu ermitteln (`model.predict()`). Außerdem werden die zuvor konfigurierten Metriken für die Testfeatures berechnet (`model.evaluate`).

Eine erste Analyse der rohen Testergebnisse zeigt deutlich schlechtere Verlustwerte und Metriken im Vergleich zu den Lernergebnissen basierend auf den Trainingsdaten und Validierungsdaten. Z. B. ist der Verlustwert (ca. $377$) deutlich größer als der Verlustwert der Validierungsdaten (ca. $46$).
Dies lässt sich dadurch erklären, dass aus den Testdaten anders als bei den Trainingsdaten keine fehlerhaften Messwerte bzw. starke Ausreißer entfernt wurden.
Diese Vermutung wird unterstützt durch den Vergleich der 90. und 95. Perzentile mit dem Maximalwert der absoluten und relativen Fehler.
Die 90. Perzentile des absoluten und relativen Fehlers der Testvorhersagen liegen z. B. bei $313,915$ bzw. $0,362$. Das zeigt, dass das Modell 90% der Testdaten mit einem geringen Fehler vorhersagen konnte. Die jeweiligen Maximalwerte zeigen jedoch sehr deutliche Abweichungen von den Perzentilen, was auf Ausreißer in den Testfehlern hinweist.
Eine Visualisierung der relativen Testfehler bestätigt, dass der größte Teil der Fehler sehr gering ist, es jedoch einige Ausreißer gibt.
Dadurch wird das arithmetische Mittel der absoluten und relativen Testfehler stark verzerrt.

Wie bereits in Abs. [-@sec:data-preparation] beschrieben, wurden bei der Datenvorbereitung Ausreißer aus den Trainingsdaten herausgefiltert, um eine möglichst gute Vorhersage für den Großteil der möglichen Daten treffen zu können -- mit dem Kompromiss, dass für wenige Daten die Vorhersagen deutliche Fehler aufweisen. Vor diesem Hintergrund werden die Testergebnisse weiter analysiert. Dazu werden 10% der Testergebnisse mit den höchsten relativen Fehlern herausgefiltert, um die Mehrheit der Testergebnisse detaillierter zu untersuchen.

Die Statistik der gefilterten Testergebnisse zeigt bereits einen deutlich besseren Mittelwert der absoluten und relativen Fehler, die sehr ähnlich zu den Werten der Trainingsdaten liegen ($47,590$ bzw. $0,117$).
Die Statistik zeigt:

- 95% der (gefilterten) Testdaten werden mit einem absoluten Fehler von $119$ oder weniger vorhergesagt
- 95% der (gefilterten) Testdaten werden mit einem relativen Fehler von $28,5\%$ oder weniger vorhergesagt

Die Visualisierung der gefilterten Testergebnisse in Streudiagramm und Histogrammen veranschaulicht, dass der größte Teil der Testdaten mit geringem Fehler vorhergesagt werden kann:

![Streudiagramm: relative Testfehler](../assets/scatter-relative-error.png){width=60%}

<!-- ![Histogramm: absolute Testfehler](../assets/hist-absolute-error.png){width=60%} -->

![Histogramm: relative Testfehler](../assets/hist-relative-error.png){width=60%}

\newpage

# Fazit und Ausblick

Für die Lernaufgabe und die gegebenen Testdaten wurde ein passendes Lernverfahren entwickelt. Dazu wurden verschiedene Konfigurationen eines neuronalen Netzes verglichen, wie die Anzahl und Größe der verborgenen Schichten und die gewählte Verlustfunktion. Basierend auf diesen Ergebnissen wurde ein Modell ausgewählt, welches für einen Großteil der Testdaten verhältnismäßig genaue Vorhersagen für die zurückgelegten Höhenmeter treffen kann.

Für eine Verbesserung des Modells sind mehr Datensätze von höherer Qualität und größerer Vielfalt notwendig. So gibt es in dem vorliegenden Datensatz bspw. wenige Datensätze für Strecken, bei denen der Fahrer eine hohe Anzahl an Höhenmetern zurückgelegt hat. Das bedeutet, dass das Modell nicht ausführlich genug lernen kann, welche Faktoren eine hohe Anzahl an Höhenmetern bewirken. In diesem Bereich werden die Vorhersagewerte eher ungenau sein. 

Weitere Verbesserungsmöglichkeiten für die Konfigurationsoptionen des Modells sind bspw. die Auswahl eines anderen Optimierers, einer anderen Aktivierungsfunktion oder einer anderen Lernrate.
