import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 1. DATOS DE ENTRADA
# ==========================================================

X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
], dtype=float)


# ==========================================================
# 2. COMPUERTAS LÓGICAS
# ==========================================================

compuertas = {
    "AND": np.array([0, 0, 0, 1], dtype=float),
    "OR":   np.array([0, 1, 1, 1], dtype=float),
    "NAND": np.array([1, 1, 1, 0], dtype=float),
    "NOR":  np.array([1, 0, 0, 0], dtype=float)
}


# ==========================================================
# 3. VALORES DE MU
# ==========================================================

mus = [0.10, 0.25, 0.50, 0.75]

epocas_maximas = 500

tolerancia = 0.001


# ==========================================================
# 4. ENTRENAMIENTO DLMS
# ==========================================================

def entrenar_dlms(
    X,
    d,
    mu,
    epocas_maximas,
    tolerancia,
    pesos_iniciales
):

    # Copiar los pesos iniciales
    w = pesos_iniciales.copy()

    historial_error = []

    historial_error2 = []

    historial_salidas = []

    convergio = False

    epoca_convergencia = epocas_maximas

    for epoca in range(epocas_maximas):

        errores = []

        errores2 = []

        for i in range(len(X)):

            x = X[i]

            d_i = d[i]

            # ----------------------------------------------
            # SALIDA
            # ----------------------------------------------

            y = np.dot(w, x)

            # ----------------------------------------------
            # ERROR
            # ----------------------------------------------

            error = d_i - y

            error2 = error ** 2

            errores.append(error)

            errores2.append(error2)

            # ----------------------------------------------
            # ACTUALIZACIÓN DLMS
            # ----------------------------------------------

            w = w + mu * error * x

        # ----------------------------------------------
        # ERROR PROMEDIO
        # ----------------------------------------------

        error_promedio = np.mean(
            np.abs(errores)
        )

        # ----------------------------------------------
        # MSE
        # ----------------------------------------------

        mse = np.mean(
            errores2
        )

        historial_error.append(
            error_promedio
        )

        historial_error2.append(
            mse
        )

        # ----------------------------------------------
        # SALIDAS ACTUALES
        # ----------------------------------------------

        salidas = np.dot(
            X,
            w
        )

        historial_salidas.append(
            salidas.copy()
        )

        # ----------------------------------------------
        # VERIFICAR CONVERGENCIA
        # ----------------------------------------------

        if mse <= tolerancia:

            convergio = True

            epoca_convergencia = epoca + 1

            break

    return {

        "pesos_iniciales":
            pesos_iniciales.copy(),

        "pesos_finales":
            w.copy(),

        "error":
            np.array(historial_error),

        "error2":
            np.array(historial_error2),

        "salidas":
            np.array(historial_salidas),

        "convergio":
            convergio,

        "epoca_convergencia":
            epoca_convergencia
    }


# ==========================================================
# 5. GENERAR PESOS INICIALES
#    SOLO CAMBIAN CUANDO CAMBIA LA COMPUERTA
# ==========================================================

pesos_por_compuerta = {}


for indice_compuerta, nombre in enumerate(compuertas):

    # Semilla diferente para cada compuerta

    semilla = 100 + indice_compuerta

    rng = np.random.default_rng(
        semilla
    )

    pesos_por_compuerta[nombre] = rng.uniform(
        low=-0.5,
        high=0.5,
        size=X.shape[1]
    )


# ==========================================================
# 6. ENTRENAMIENTO
# ==========================================================

resultados = {}


for nombre, deseada in compuertas.items():

    resultados[nombre] = {}

    # Los mismos pesos iniciales se utilizan
    # para todos los valores de μ de la compuerta

    pesos_iniciales = (
        pesos_por_compuerta[nombre]
    )

    for mu in mus:

        resultado = entrenar_dlms(
            X,
            deseada,
            mu,
            epocas_maximas,
            tolerancia,
            pesos_iniciales
        )

        resultados[nombre][mu] = resultado


# ==========================================================
# 7. PRIMERAS 16 GRÁFICAS
#    ERROR PROMEDIO + ERROR CUADRÁTICO + TOLERANCIA
# ==========================================================

for nombre, deseada in compuertas.items():

    for mu in mus:

        resultado = resultados[nombre][mu]

        error = resultado["error"]

        error2 = resultado["error2"]

        salidas = resultado["salidas"]

        epocas = np.arange(
            1,
            len(error) + 1
        )


        # ==================================================
        # DATOS FINALES
        # ==================================================

        pesos_iniciales = (
            resultado["pesos_iniciales"]
        )

        pesos_finales = (
            resultado["pesos_finales"]
        )

        salida_final = (
            salidas[-1]
        )

        salida_binaria = (
            salida_final >= 0.5
        ).astype(int)

        error_final = (
            error[-1]
        )

        mse_final = (
            error2[-1]
        )


        # ==================================================
        # RESULTADO DE CONVERGENCIA
        # ==================================================

        if resultado["convergio"]:

            estado = (
                f"CONVERGIÓ en la época "
                f"{resultado['epoca_convergencia']}"
            )

        else:

            estado = (
                f"NO alcanzó la tolerancia de "
                f"{tolerancia} en "
                f"{epocas_maximas} épocas"
            )


        # ==================================================
        # MOSTRAR TABLA COMO TEXTO
        # ==================================================

        print("\n")

        print("=" * 100)

        print(
            f"COMPUERTA: {nombre}   |   μ = {mu}"
        )

        print("=" * 100)

        print(
            f"{'DATO':<35} | VALOR"
        )

        print("-" * 100)

        print(
            f"{'Pesos iniciales aleatorios':<35} | "
            f"{np.array2string(np.round(pesos_iniciales, 5))}"
        )

        print(
            f"{'Pesos finales':<35} | "
            f"{np.array2string(np.round(pesos_finales, 5))}"
        )

        print(
            f"{'Salida deseada':<35} | "
            f"{np.array2string(deseada.astype(int))}"
        )

        print(
            f"{'Salida aproximada':<35} | "
            f"{np.array2string(np.round(salida_final, 5))}"
        )

        print(
            f"{'Salida binaria':<35} | "
            f"{np.array2string(salida_binaria)}"
        )

        print(
            f"{'Error final':<35} | "
            f"{error_final:.8f}"
        )

        print(
            f"{'Error² / MSE final':<35} | "
            f"{mse_final:.8f}"
        )

        print(
            f"{'Resultado':<35} | "
            f"{estado}"
        )

        print("=" * 100)


        # ==================================================
        # CREAR GRÁFICA
        # ==================================================

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )


        # ==================================================
        # ERROR PROMEDIO
        # ==================================================

        ax.plot(
            epocas,
            error,
            linewidth=2,
            label="|Error| promedio"
        )


        # ==================================================
        # ERROR CUADRÁTICO / MSE
        # ==================================================

        ax.plot(
            epocas,
            error2,
            linewidth=2,
            label="Error² / MSE"
        )


        # ==================================================
        # LÍNEA DE TOLERANCIA
        # ==================================================

        ax.axhline(
            y=tolerancia,
            linestyle="--",
            linewidth=1.5,
            label=f"Tolerancia = {tolerancia}"
        )


        # ==================================================
        # MARCAR CONVERGENCIA
        # ==================================================

        if resultado["convergio"]:

            ec = (
                resultado["epoca_convergencia"]
            )

            ax.scatter(
                ec,
                error2[ec - 1],
                s=80,
                zorder=5,
                label=f"Convergencia = {ec}"
            )


        # ==================================================
        # CONFIGURACIÓN DE LA GRÁFICA
        # ==================================================

        ax.set_xlabel(
            "Época",
            fontsize=11
        )

        ax.set_ylabel(
            "Error",
            fontsize=11
        )

        ax.set_title(
            f"Descenso del error - {nombre} - μ = {mu}",
            fontsize=15,
            fontweight="bold"
        )

        ax.grid(True)

        ax.legend()


        # ==================================================
        # MOSTRAR GRÁFICA
        # ==================================================

        plt.tight_layout()

        plt.show()


# ==========================================================
# 8. TABLA RESUMEN FINAL
# ==========================================================

print("\n")

print("=" * 100)

print(
    "TABLA RESUMEN"
)

print("=" * 100)

print(
    f"{'Compuerta':<12}"
    f"{'μ':<10}"
    f"{'Épocas':<12}"
    f"{'Error':<15}"
    f"{'MSE':<15}"
    f"{'Convergencia':<15}"
)

print("-" * 100)


for nombre in compuertas:

    for mu in mus:

        resultado = resultados[nombre][mu]

        numero_epocas = len(
            resultado["error"]
        )

        estado = (
            "Sí"
            if resultado["convergio"]
            else "No"
        )

        print(
            f"{nombre:<12}"
            f"{mu:<10}"
            f"{numero_epocas:<12}"
            f"{resultado['error'][-1]:<15.8f}"
            f"{resultado['error2'][-1]:<15.8f}"
            f"{estado:<15}"
        )


print("=" * 100)


# ==========================================================
# 9. RECOMENDACIÓN DEL NÚMERO DE ÉPOCAS
# ==========================================================

epocas_convergencia = []


for nombre in compuertas:

    for mu in mus:

        resultado = resultados[nombre][mu]

        if resultado["convergio"]:

            epocas_convergencia.append(
                resultado["epoca_convergencia"]
            )


print("\n")

print("=" * 100)

print(
    "ANÁLISIS DEL NÚMERO DE ÉPOCAS"
)

print("=" * 100)


if len(epocas_convergencia) > 0:

    mayor_convergencia = max(
        epocas_convergencia
    )

    print(
        "Mayor época de convergencia encontrada:",
        mayor_convergencia
    )

else:

    mayor_convergencia = 0

    print(
        "Ninguna combinación alcanzó la tolerancia."
    )


# ==========================================================
# REVISAR SI EXISTEN CASOS QUE NO CONVERGIERON
# ==========================================================

casos_no_convergentes = 0


for nombre in compuertas:

    for mu in mus:

        resultado = resultados[nombre][mu]

        if not resultado["convergio"]:

            casos_no_convergentes += 1


# ==========================================================
# RECOMENDACIÓN
# ==========================================================

if casos_no_convergentes > 0:

    mejor_numero_epocas = epocas_maximas

    print(
        "\nNúmero de épocas recomendado para este experimento:"
    )

    print(
        f"→ {mejor_numero_epocas} épocas"
    )

    print(
        "\nMotivo:"
    )

    print(
        "Existen combinaciones de compuerta y μ "
        "que no alcanzan la tolerancia de "
        f"{tolerancia} dentro del número de épocas "
        "menor establecido."
    )

    print(
        f"Por ello, se recomienda conservar "
        f"{epocas_maximas} épocas como máximo."
    )

else:

    mejor_numero_epocas = mayor_convergencia

    print(
        "\nNúmero de épocas recomendado para este experimento:"
    )

    print(
        f"→ {mejor_numero_epocas} épocas"
    )

    print(
        "\nTodas las combinaciones alcanzaron "
        "la tolerancia establecida."
    )


print("=" * 100)