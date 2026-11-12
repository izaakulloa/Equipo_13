# Instrucciones de Ejecución - Proyecto Integrador: Regresión

Este documento contiene las instrucciones necesarias para la correcta ejecución y reproducción del script unificado (`Script_terminado.R`), correspondiente al modelo de regresión para la predicción del precio de vehículos usados.

## 1. Requisitos Previos

*   **Entorno de Ejecución:** RStudio.
*   **Gestión del Directorio:** Es fundamental abrir el proyecto mediante el archivo `codigo.Rproj`. Esto fijará el directorio de trabajo a la carpeta actual de manera automática y evitará errores de rutas de archivo (Path Errors).
*   **Ubicación de los Datos:** El dataset (`used_cars.csv`) debe permanecer en el directorio `datos` ubicado un nivel superior a la carpeta `codigo` (ruta relativa: `../datos/used_cars.csv`), o en su defecto, en la misma carpeta del script. El código incluye verificación de rutas relativas.

## 2. Instalación de Dependencias

El script está configurado para verificar e instalar automáticamente cualquier paquete faltante al inicio de su ejecución. Las dependencias principales incluyen:
`tidyverse`, `caret`, `janitor`, `rpart`, `rpart.plot`, `Metrics`, `lmtest`, `car`, `glmnet`, `mgcv`, `caretEnsemble`, `ranger`, `xgboost`.

No se requiere instalación manual previa. 

## 3. Procedimiento de Ejecución

1.  Haga doble clic en el archivo `codigo.Rproj` para iniciar RStudio en el entorno del proyecto.
2.  Abra el archivo `Script_terminado.R`.
3.  Reinicie la sesión de R para asegurar un entorno limpio (`Session > Restart R`).
4.  Ejecute el script completo (Ctrl + Alt + R o el botón `Source`).

## 4. Estructura y Flujo de los Datos (Prevención de Fuga de Información)

El código ha sido estructurado siguiendo estrictamente las directrices metodológicas:
*   La partición 80/20 se realiza en las líneas iniciales, aislando el conjunto `test` antes de aplicar cualquier regla de imputación o estandarización.
*   Los 5 folds de validación cruzada están centralizados mediante la semilla `123` y el objeto compartido `folds`, garantizando que todos los modelos evalúen las mismas observaciones.
*   El preprocesamiento (escalado, centrado y remoción de varianza cero) se computa de forma interna en cada fold para asegurar ausencia total de *Data Leakage*.

## 5. Salidas Generadas (Archivos Exportados)

Tras una ejecución exitosa, el script generará automáticamente los siguientes 6 archivos en la carpeta de trabajo:

**Diagnósticos Visuales:**
*   `diagnostico_OLS.pdf`: Evaluación de supuestos del modelo de regresión lineal.
*   `qq_residuos_OLS.pdf`: Gráfico de cuantiles teóricos vs cuantiles muestrales.
*   `diagnostico_GAM.pdf`: Efectos parciales de suavizado del modelo ganador.

**Métricas y Resultados Finales:**
*   `tabla_cv_modelos_individuales.csv`: Registro del desempeño durante el entrenamiento.
*   `tabla_comparativa_validacion.csv`: Comparación oficial de RMSE, MAE y $R^2$ ordenados para selección (con base exclusivamente en datos de entrenamiento).
*   `resultado_test_final.csv`: Evaluación única y definitiva del modelo ganador contra el baseline en el conjunto de prueba aislado.