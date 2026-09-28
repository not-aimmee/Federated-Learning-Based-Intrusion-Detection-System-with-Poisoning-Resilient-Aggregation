# Checkpoint 1 - Team Ready (Week 4, owner: All)

Deliverable: the team has learned the core concepts, and the CICIDS2017 dataset is acquired and understood.
This page is the written record. Tick the checklist at the bottom together as a team.

## 1. Core concepts (one-paragraph versions)

**Intrusion Detection System (IDS).** Software that inspects network traffic and flags malicious activity.
*Signature-based* IDS matches known attack patterns; *anomaly / ML-based* IDS learns what normal traffic looks
like (or learns labelled attack patterns) and can generalise to variants. This project builds an ML-based,
flow-level IDS: each network *flow* (one connection) is a row of ~78 statistics (duration, packet counts,
inter-arrival times, flag counts ...) and the model outputs BENIGN or an attack class.

**Federated Learning (FL).** Several organisations train ONE shared model without pooling their raw data.
Each round: (1) the server sends the current global model to the clients; (2) each client trains it locally on
its own data; (3) clients send back only model parameters; (4) the server aggregates them (FedAvg = weighted
average by number of local examples) into the next global model. Raw traffic never leaves an organisation - the
privacy argument for using FL in security, where traffic logs are sensitive.

**Why this needs care.** (a) *Non-IID data*: real organisations see different traffic mixes, which slows and
destabilises training (simulated here with `partition-alpha`, a Dirichlet split over classes). (b) *Class
imbalance*: attacks are rare, so accuracy is misleading - always report balanced accuracy and macro-F1.
(c) *Trust*: the server must trust every client's update - the Stage 2 problem (poisoning).

**Security basics used later (Stage 2).** Data poisoning (label flipping), model poisoning (sending a crafted
update), and robust aggregation (median, trimmed mean, Krum/Multi-Krum) that limits what a minority of
malicious clients can do. Already implemented and unit-tested in `sklearnexample/aggregators.py` and
`attacks.py`; Stage 1 runs with no attackers.

## 2. Dataset - CICIDS2017 (Canadian Institute for Cybersecurity)

- 5 days of captured traffic (Mon-Fri), 8 CSV files (the "MachineLearningCVE" release), ~2.83 M flows,
  78 numeric features + `Label`; BENIGN plus 14 attack types (DoS/DDoS variants, PortScan, Brute-force,
  Web attacks, Infiltration, Botnet, Heartbleed).
- Known data-quality issues (already handled in `legacy/data_inspection*.py` and `scripts/preprocess_cicids2017.py`):
  column names with leading spaces, `inf`/`NaN` in `Flow Bytes/s` and `Flow Packets/s`, negative flow durations,
  many exact duplicate rows, and mangled characters in web-attack labels.
- Extremely imbalanced: BENIGN is ~80 % of flows; several attacks (Heartbleed, Infiltration, SQL injection)
  have fewer than ~40 samples. This is why accuracy alone looks great (~99 %) while macro-F1 does not.
- **Leakage finding from the team's own work:** identical feature vectors (some with conflicting labels) appear in both
  train and test after a plain random split, which inflates scores. The *group-aware split*
  (`data/splits/*_group_aware.csv`) fixes this and is the split every Stage 1 model must use.

## 3. Where the data lives (never committed to git)

```
data/raw/        the 8 original CICIDS2017 CSVs
data/splits/     train/validation/test(_group_aware).csv   <- Checkpoint 2 output (TM2)
data/processed/  X_*.npy, y_*.npy, meta.json               <- built by scripts/prepare_from_splits.py
```

## 4. Checkpoint 1 sign-off checklist

- [ ] Every member can explain the FL round loop (steps 1-4 above) without notes.
- [ ] Every member can explain why accuracy is misleading here and which metrics we report instead.
- [ ] CICIDS2017 CSVs are in `data/raw/`; row/label counts match the published dataset.
- [ ] Everyone has run `pytest` and the synthetic quick-start from the README successfully.
- [ ] Everyone has read `docs/notes/observations.txt` and the Flower tutorial series linked in it.
