# Verifica que se reciban exactamente tres parámetros
LC_NUMERIC=C

if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <Num_bloques> <Tam_bloque>"
    exit 1
fi

# Almacena los argumentos en variables para mayor claridad
NUM_BLOQUES=$1
TAM_BLOQUE=$2

# Ejecuta el programa de CPU y guarda la salida en un archivo temporal
./transformacion-cpu $NUM_BLOQUES $TAM_BLOQUE > cpu_output.txt

# Imprime las últimas tres líneas de la salida del programa de CPU
echo "***********"
echo "Salida de transformacion-cpu:"
tail -n 5 cpu_output.txt

# Extrae el tiempo de ejecución del programa de CPU
CPU_TIME=$(tail -n 2 cpu_output.txt | awk -F '= ' '{print $NF}')

# Ejecuta el programa de GPU y guarda la salida en un archivo temporal
./transformacion-gpu $NUM_BLOQUES $TAM_BLOQUE '1' > gpu_output_shared.txt
./transformacion-gpu $NUM_BLOQUES $TAM_BLOQUE '0' > gpu_output_no_shared.txt

# Imprime las últimas tres líneas de la salida del programa de GPU
echo "***********"
echo "Salida de transformacion-gpu-shared:"
tail -n 7 gpu_output_shared.txt
echo "***********"
echo "Salida de transformacion-gpu-shared:"
tail -n 7 gpu_output_no_shared.txt

# Extrae el tiempo de ejecución del programa de GPU
GPU_TIME_SHARED=$(tail -n 2 gpu_output_shared.txt | awk -F '= ' '{print $NF}')
GPU_TIME_NO_SHARED=$(tail -n 2 gpu_output_no_shared.txt | awk -F '= ' '{print $NF}')

# Calcula la relación entre los tiempos de ejecución de CPU y GPU
SPEEDUP_SHARED=$(echo "scale=6; $CPU_TIME / $GPU_TIME_SHARED" | LC_NUMERIC=C bc -l)
SPEEDUP_NO_SHARED=$(echo "scale=6; $CPU_TIME / $GPU_TIME_NO_SHARED" | LC_NUMERIC=C bc -l)

# Imprime la relación de tiempo de ejecución
echo "***********"
echo "NUM_BLOQUES: $NUM_BLOQUES, TAM_BLOQUE: $TAM_BLOQUE"
echo "CPU_TIME: $CPU_TIME, GPU_TIME_SHARED: $GPU_TIME_SHARED, GPU_TIME_NO_SHARED: $GPU_TIME_NO_SHARED"
printf "Speedup CPU/GPU_SHARED es: %.6f\n" $SPEEDUP_SHARED
printf "Speedup CPU/GPU_NO_SHARED es: %.6f\n" $SPEEDUP_NO_SHARED
echo ""

# Elimina los archivos temporales de salida
rm cpu_output.txt gpu_output_shared.txt gpu_output_no_shared.txt
