# Semilla LLM

Una semilla mínima de LLM en Python: un pequeño Transformer autoregresivo entrenable con texto propio.

Este proyecto no crea un ChatGPT completo. Es una base educativa para entender cómo un modelo de lenguaje aprende a predecir el siguiente token.

## Estructura

```text
semilla_llm/
├── data/
│   └── corpus.txt
├── src/
│   ├── model.py
│   └── tokenizer.py
├── train.py
├── generate.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenar

Edita `data/corpus.txt` con tu propio texto y ejecuta:

```bash
python train.py
```

El entrenamiento guardará un modelo en:

```text
checkpoints/semilla_llm.pt
```

## Generar texto

```bash
python generate.py --prompt "El infinito es"
```

También puedes cambiar temperatura y cantidad de caracteres:

```bash
python generate.py --prompt "El lenguaje" --max-new-tokens 300 --temperature 0.8
```

## Subir a GitHub

```bash
git init
git add .
git commit -m "Primera semilla de LLM"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/semilla-llm.git
git push -u origin main
```

## Idea filosófica

Esta semilla aprende una cosa muy simple:

> dado un fragmento de texto, intenta predecir qué viene después.

Desde ahí nace la ilusión del lenguaje: no como conocimiento pleno, sino como continuidad probable.
