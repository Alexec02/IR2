import csv

input_file = 'rotation_memory.csv'
output_file = 'datos_corregidos.csv'

def corregir_acciones(accion_str):
    # Quitar los corchetes si están
    accion_str = accion_str.strip('[]')
    # Separar por espacio, agregar coma
    valores = accion_str.split()
    return f"[{', '.join(valores)}]"

with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Escribir encabezado
    encabezado = next(reader)
    writer.writerow(encabezado)

    for row in reader:
        # Asume que la columna 2 es la acción
        row[2] = corregir_acciones(row[2])
        writer.writerow(row)

print("Archivo corregido guardado como", output_file)
