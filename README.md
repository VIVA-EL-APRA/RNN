# Traductor RNN - UAC

Traductor automático Inglés <-> Español basado en Redes Neuronales Recurrentes (RNN) implementado con la metodología CRISP-ML(Q).

## Características

- **Arquitectura**: Seq2Seq LSTM with Attention
- **BLEU Score**: 0.90
- **Épocas**: 100
- **Parámetros**: 7,697,741

## Uso

### HuggingFace Spaces (Web)
Accede a: https://huggingface.co/spaces/NICOMOSHE/RNN

### Local (Python)
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar interfaz
py app.py
```

### Google Colab
Abre el notebook y ejecuta las celdas.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `train_v3.py` | Entrenamiento del modelo RNN |
| `app.py` | Interfaz Gradio para producción |
| `translator.pt` | Modelo entrenado |
| `notebook.py` | Curvas de pérdida |

## CRISP-ML(Q)

El proyecto sigue las 6 fases de CRISP-ML(Q):

1. **Business Understanding** - Métricas: BLEU ≥ 0.30
2. **Data Preparation** - Corpus de 322 parejas
3. **Modeling** - Seq2Seq con LSTM
4. **Evaluation** - BLEU Score = 0.90
5. **Deployment** - API REST + Interfaz
6. **Monitoring** - Detección de deriva

## Ejemplos

| Inglés | Español |
|--------|----------|
| hello | hola |
| thank you | gracias |
| i am a student | soy estudiante |
| where is the library | donde está la biblioteca |

## Créditos

Desarrollado para el Taller 2.5 - Proyecto RNN
Universidad Autónoma del Caribe (UAC)