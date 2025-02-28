Gruppenmitglieder:
1. Mitglied: Enes Yilmaz (30336049), yilmaz.enes@fh-swf.de
2. Mitglied: Kiryl Hlebau (30335828), hlebau.kiryl@fh-swf.de
3. Mitglied: Dennis Waltemathe (30331833), waltemathe.dennis@fh-swf.de
4. Mitglied: Demir Dolovac (30328488), dolovac.demir@fh-swf.de

Alle Ordner haben eine ähnliche Struktur:

agent.py: Enthält den Agenten, der das Spiel spielt, sowie main(). Ruft environment und model auf.
environment.py: Stellt die Spielumgebung dar, kompiliert das Spiel und gibt Rewards an den Agenten zurück.
model.py: Enthält das trainierbare Modell, das die Aktionen bestimmt.
play_game_AI.py: Ermöglicht es dem trainierten Modell, das Spiel zu spielen.
!!!
IM FOLDER Double_QLearning_abgabe und QLearning_abgabe GIBT ES KEIN model.py, DORT SIND DIE Q-TABLES IN agent.py DEFINIERT
UND DIE MODELLE WERDEN IM GLEICHEN VERZEICHNIS ALS npy-DATEI GESPEICHERT
!!!

Zusätzlich gibt es gemeinsame Ordner:

assets: Bilder für die Umgebung.
helper: Klasse mit Methoden zur Score-Visualisierung.

!!!PER BEFEHL python agent.py BEGINNT DAS TRAINING DES AGENTEN!!!

Arbeitsaufteilung:
Enes Yilmaz: enviroment.py, DQN + SARSA-Variante
Dennis Waltemathe: DQN + SARSA-Variante
Kiryl Hlebau: DQN
Demir Dolovac: Q-Learning, Pixil (Design)

Gemeinsam: agent.py bearbeitet, Ausarbeitung


