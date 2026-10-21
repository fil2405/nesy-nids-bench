# server/

Script per lanciare il benchmark su una macchina senza accesso interattivo. Ordine e diagramma in `WORKFLOW.md`.

    bash server/build_package.sh        # portatile, checkout pulito
    bash install.sh                     # server, dentro il pacchetto
    bash server/00_smoke.sh
    bash server/01_scaling.sh
    bash server/02_tune.sh DATASET      # NESY_JOBS_NEURAL / NESY_JOBS_CLASSIC per il parallelo
    bash server/03_final.sh DATASET
    bash server/04_stress.sh E7
    bash server/05_collect.sh NOME      # tar.gz da restituire

`python server/plan.py table` mostra cosa è pronto e cosa manca. Log e avanzamento in `results/server/`.
