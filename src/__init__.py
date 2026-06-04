"""Paquete del sistema de reconocimiento facial con LBP.

Contiene los módulos del pipeline:

    image_loader -> lbp -> features -> (similarity, model) -> evaluate

La idea pedagógica es que cada módulo tenga una sola responsabilidad y se pueda
leer de forma independiente.
"""
