class SistemaExpertoComida:
    def __init__(self, ingredientes):
        self.ingredientes = ingredientes

    def sugerir_plato(self):
        if 'pollo' in self.ingredientes and 'arroz' in self.ingredientes:
            return 'Puedes cocinar pollo con arroz.'
        elif 'pasta' in self.ingredientes and 'salsa de tomate' in self.ingredientes:
            return 'Puedes cocinar pasta con salsa de tomate.'
        elif 'lechuga' in self.ingredientes and 'tomate' in self.ingredientes:
            return 'Puedes preparar una ensalada.'
        elif 'carne' in self.ingredientes and 'papas' in self.ingredientes:
            return 'Puedes hacer carne con papas.'
        else:
            return 'Lo siento, no tienes suficientes ingredientes para hacer ningún plato sugerido.'

# Ejemplo de uso
ingredientes_disponibles = ['pollo', 'arroz', 'verduras']
sistema_experto = SistemaExpertoComida(ingredientes_disponibles)
print(sistema_experto.sugerir_plato())