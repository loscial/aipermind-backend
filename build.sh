#!/usr/bin/env bash
# Esce in caso di errore
set -o errexit

# Forza l'aggiornamento di pip e l'installazione di tutte le dipendenze
pip install --upgrade pip
pip install -r requirements.txt